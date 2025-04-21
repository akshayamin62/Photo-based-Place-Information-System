import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol
import requests
from io import BytesIO
import pickle

# Feature Extraction MRJob class
class ExtractFeaturesMRJob(MRJob):
    INPUT_PROTOCOL = JSONProtocol
    OUTPUT_PROTOCOL = JSONProtocol
    
    def configure_args(self):
        super(ExtractFeaturesMRJob, self).configure_args()
        self.add_file_arg('--model', help='Path to the pretrained model')
        self.add_file_arg('--transform_params', help='Path to transformation parameters')
    
    def load_model(self):
        # Load the model on each worker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()  # Remove classification layer
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load model weights if provided
        if self.options.model and os.path.exists(self.options.model):
            self.model.load_state_dict(torch.load(self.options.model, map_location=self.device))
            
        # Setup transformation
        if self.options.transform_params and os.path.exists(self.options.transform_params):
            with open(self.options.transform_params, 'rb') as f:
                params = pickle.load(f)
            self.transform = transforms.Compose([
                transforms.Resize(params.get('resize', 256)),
                transforms.CenterCrop(params.get('crop', 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=params.get('mean', [0.485, 0.456, 0.406]), 
                    std=params.get('std', [0.229, 0.224, 0.225])
                )
            ])
        else:
            # Default transformation
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def mapper_init(self):
        self.load_model()
    
    def mapper(self, _, record):
        # Extract features from a single image
        try:
            place_id = record['place_id']
            img_path = record['image_path']
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    feature = self.model(img_tensor).cpu().numpy()[0]
                
                yield place_id, {'place_id': place_id, 'features': feature.tolist()}
            else:
                yield place_id, {'place_id': place_id, 'features': None, 'error': 'File not found'}
                
        except Exception as e:
            yield str(record.get('place_id', 'unknown')), {'place_id': record.get('place_id', 'unknown'), 
                                                         'features': None, 
                                                         'error': str(e)}

# Place Info Aggregation MRJob class
class PlaceInfoMRJob(MRJob):
    INPUT_PROTOCOL = JSONProtocol
    OUTPUT_PROTOCOL = JSONProtocol
    
    def configure_args(self):
        super(PlaceInfoMRJob, self).configure_args()
        self.add_file_arg('--metadata', help='Path to place metadata CSV')
        
    def mapper_init(self):
        if self.options.metadata and os.path.exists(self.options.metadata):
            self.places_df = pd.read_csv(self.options.metadata)
            # Convert to dictionary for faster lookups
            self.places_dict = self.places_df.set_index('place_id').to_dict('index')
        else:
            self.places_dict = {}
    
    def mapper(self, place_id, _):
        # Get basic place info
        place_info = self.places_dict.get(place_id, {})
        if place_info:
            # Emit the place info and its category for reducer
            category = place_info.get('category', 'unknown')
            lat = place_info.get('latitude', 0)
            lon = place_info.get('longitude', 0)
            
            # Primary output: the place itself
            yield place_id, {'place_info': place_info}
            
            # Secondary outputs for aggregation
            yield f"category:{category}", {'related_place': {
                'place_id': place_id,
                'name': place_info.get('name', 'Unknown'),
                'category': category,
                'latitude': lat,
                'longitude': lon
            }}
            
            # Location-based grouping (simplified)
            lat_lon_key = f"location:{int(lat*10)},{int(lon*10)}"  # Group by rough location
            yield lat_lon_key, {'nearby_place': {
                'place_id': place_id,
                'name': place_info.get('name', 'Unknown'),
                'category': category,
                'latitude': lat,
                'longitude': lon,
                'distance': 0  # Will be calculated in reducer
            }}
    
    def reducer(self, key, values):
        values_list = list(values)
        
        if key.startswith('category:'):
            # For category keys, collect related places
            related_places = [v.get('related_place') for v in values_list if 'related_place' in v]
            category = key.split(':', 1)[1]
            yield category, {'related_places': related_places[:5]}  # Limit to 5
            
        elif key.startswith('location:'):
            # For location keys, collect nearby places
            nearby_places = [v.get('nearby_place') for v in values_list if 'nearby_place' in v]
            location = key.split(':', 1)[1]
            yield location, {'nearby_places': nearby_places[:5]}  # Limit to 5
            
        else:
            # For place_id keys
            place_info = next((v.get('place_info') for v in values_list if 'place_info' in v), {})
            if place_info:
                yield key, place_info

# Main class for the Place Recognition System
class PlaceRecognitionSystem:
    def __init__(self, dataset_path, model_path=None):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.load_dataset()
        
    def setup_model(self):
        # Initialize pretrained model for feature extraction
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()  # Remove classification layer
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Save transform parameters
        transform_params = {
            'resize': 256,
            'crop': 224,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
        transform_path = os.path.join(self.dataset_path, "transform_params.pkl")
        with open(transform_path, 'wb') as f:
            pickle.dump(transform_params, f)
        
        # Load model weights if provided
        if self.model_path and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
    def load_dataset(self):
        # Load place metadata
        metadata_path = os.path.join(self.dataset_path, "places_metadata.csv")
        self.places_df = pd.read_csv(metadata_path)
        
        # Load precomputed features or compute them
        feature_path = os.path.join(self.dataset_path, "place_features.npy")
        if os.path.exists(feature_path):
            self.place_features = np.load(feature_path)
            self.build_nn_index()
        else:
            self.extract_features_from_dataset()
    
    def extract_features_from_dataset(self):
        # Create input data for MRJob
        input_data = []
        for idx, row in self.places_df.iterrows():
            img_path = os.path.join(self.dataset_path, "images", row['image_filename'])
            if os.path.exists(img_path):
                input_data.append({
                    'place_id': row['place_id'],
                    'image_path': img_path
                })
        
        # Write input data to file
        input_file = os.path.join(self.dataset_path, "feature_extraction_input.json")
        with open(input_file, 'w') as f:
            for item in input_data:
                f.write(json.dumps(item) + '\n')
        
        # Run MRJob for feature extraction
        mr_job = ExtractFeaturesMRJob(
            args=[
                '--model', os.path.join(self.dataset_path, "model.pth") if self.model_path else '',
                '--transform_params', os.path.join(self.dataset_path, "transform_params.pkl"),
                input_file
            ]
        )
        
        # Collect results
        features_dict = {}
        with mr_job.make_runner() as runner:
            runner.run()
            for place_id, data in mr_job.parse_output(runner.cat_output()):
                if data.get('features') is not None:
                    features_dict[place_id] = np.array(data['features'])
        
        # Convert to numpy array in the same order as places_df
        self.place_features = np.zeros((len(self.places_df), 2048))  # ResNet50 feature size
        for i, row in self.places_df.iterrows():
            place_id = row['place_id']
            if place_id in features_dict:
                self.place_features[i] = features_dict[place_id]
        
        # Save features for future use
        np.save(os.path.join(self.dataset_path, "place_features.npy"), self.place_features)
        
        # Build nearest neighbors index
        self.build_nn_index()
    
    def build_nn_index(self):
        # Build nearest neighbors index for fast retrieval
        self.nn_index = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
        self.nn_index.fit(self.place_features)
    
    def process_image(self, image_path=None, image_url=None):
        """Process an image to identify the place and return information"""
        try:
            if image_url:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content)).convert('RGB')
            elif image_path:
                img = Image.open(image_path).convert('RGB')
            else:
                raise ValueError("Either image_path or image_url must be provided")
                
            # Extract features
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(img_tensor).cpu().numpy()[0]
                
            # Find nearest neighbors
            distances, indices = self.nn_index.kneighbors([feature], n_neighbors=5)
            
            # Get top match
            top_match_idx = indices[0][0]
            confidence = 1 - distances[0][0]  # Convert distance to confidence score
            
            # Get place info
            place_id = self.places_df.iloc[top_match_idx]['place_id']
            
            return self.get_place_info(place_id, confidence)
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_place_info(self, place_id, confidence):
        """Use MRJob to get and aggregate place information"""
        # Create input for MRJob
        input_file = os.path.join(self.dataset_path, "place_info_input.json")
        with open(input_file, 'w') as f:
            f.write(json.dumps({place_id: {}}) + '\n')
        
        # Run MRJob for place info retrieval
        mr_job = PlaceInfoMRJob(
            args=[
                '--metadata', os.path.join(self.dataset_path, "places_metadata.csv"),
                input_file
            ]
        )
        
        # Collect results
        place_info = {}
        related_places = []
        nearby_places = []
        
        with mr_job.make_runner() as runner:
            runner.run()
            for key, value in mr_job.parse_output(runner.cat_output()):
                if key == place_id:
                    place_info = value
                elif key.startswith('category:'):
                    if 'related_places' in value:
                        related_places = value['related_places']
                elif key.startswith('location:'):
                    if 'nearby_places' in value:
                        nearby_places = value['nearby_places']
        
        # Format output
        result = {
            "place_id": place_id,
            "place_name": place_info.get('name', 'Unknown'),
            "location": {
                "latitude": float(place_info.get('latitude', 0)), 
                "longitude": float(place_info.get('longitude', 0))
            },
            "category": place_info.get('category', 'Unknown'),
            "description": place_info.get('description', ''),
            "confidence": float(confidence),
            "related_places": related_places[:3],  # Limit to 3
            "nearby_places": nearby_places[:3]     # Limit to 3
        }
        
        return result

# Function to prepare dataset structure
def prepare_dataset_structure(base_path):
    """Creates necessary folders and files for the dataset"""
    os.makedirs(os.path.join(base_path, "images"), exist_ok=True)
    
    # Create a sample metadata file if it doesn't exist
    metadata_path = os.path.join(base_path, "places_metadata.csv")
    if not os.path.exists(metadata_path):
        sample_data = pd.DataFrame({
            'place_id': ['place_001', 'place_002', 'place_003'],
            'name': ['Eiffel Tower', 'Taj Mahal', 'Grand Canyon'],
            'latitude': [48.8584, 27.1751, 36.1069],
            'longitude': [2.2945, 78.0421, -112.1129],
            'category': ['landmark', 'landmark', 'natural'],
            'description': [
                'Famous iron tower in Paris, France',
                'Iconic marble mausoleum in Agra, India',
                'Steep-sided canyon in Arizona, USA'
            ],
            'image_filename': ['eiffel.jpg', 'taj_mahal.jpg', 'grand_canyon.jpg']
        })
        sample_data.to_csv(metadata_path, index=False)
        print(f"Created sample metadata file at {metadata_path}")
    
    return base_path

# Example usage
if __name__ == "__main__":
    # Prepare dataset structure
    dataset_path = prepare_dataset_structure("./place_dataset")
    
    # Initialize the system
    system = PlaceRecognitionSystem(dataset_path=dataset_path)
    
    # Process a place image
    result = system.process_image(image_path="test_image.jpg")
    print(json.dumps(result, indent=2))
    
    # To run MRJob directly:
    # python place_recognition.py --metadata=./place_dataset/places_metadata.csv ./place_dataset/place_info_input.json