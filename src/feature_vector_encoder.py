# File: src/feature_vector_encoder.py
import numpy as np
from typing import Dict, List
import pandas as pd
from hierarchical_segmentation import HierarchicalBodySegmentation
from geometric_features import GeometricFeatureExtractor

class HierarchicalFeatureEncoder:
    def __init__(self):
        self.feature_names = []
        self.scaler = None
        
    def create_hierarchical_features(self, keypoints: np.ndarray) -> np.ndarray:
        """Create structured hierarchical feature vector"""
        segmentation = HierarchicalBodySegmentation()
        feature_extractor = GeometricFeatureExtractor(segmentation)
        
        # Extract different types of features
        joint_angles = feature_extractor.extract_joint_angles(keypoints)
        proportions = feature_extractor.extract_limb_proportions(keypoints)  
        spatial_relations = feature_extractor.extract_spatial_relationships(keypoints)
        
        # Combine all features
        all_features = {**joint_angles, **proportions, **spatial_relations}
        
        # Convert to ordered feature vector
        feature_vector = []
        feature_names = []
        
        for feature_name, value in all_features.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                feature_vector.append(value)
                feature_names.append(feature_name)
        
        self.feature_names = feature_names
        return np.array(feature_vector)
    
    def encode_pose_sequence(self, keypoints_sequence: List[np.ndarray]) -> np.ndarray:
        """Encode sequence of poses into feature matrix"""
        feature_matrix = []
        
        for keypoints in keypoints_sequence:
            if keypoints is not None:
                features = self.create_hierarchical_features(keypoints)
                if len(features) > 0:
                    feature_matrix.append(features)
        
        return np.array(feature_matrix) if feature_matrix else np.array([])
    
    def get_feature_importance_map(self) -> Dict[str, List[str]]:
        """Map features to body parts for interpretability"""
        feature_map = {
            'joint_angles': [name for name in self.feature_names if 'elbow' in name or 'knee' in name or 'shoulder' in name or 'hip' in name],
            'body_proportions': [name for name in self.feature_names if 'ratio' in name or 'width' in name or 'length' in name],
            'spatial_relations': [name for name in self.feature_names if 'distance' in name or 'extension' in name]
        }
        return feature_map

