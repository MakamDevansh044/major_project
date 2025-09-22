# File: src/geometric_features.py
import numpy as np
import math
from typing import Dict, List, Tuple
from hierarchical_segmentation import HierarchicalBodySegmentation

class GeometricFeatureExtractor:
    def __init__(self, segmentation: HierarchicalBodySegmentation):
        self.segmentation = segmentation
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at point p2 formed by p1-p2-p3"""
        # Create vectors
        v1 = p1[:2] - p2[:2]  # Use only x, y coordinates
        v2 = p3[:2] - p2[:2]
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = math.acos(cos_angle)
        
        return math.degrees(angle)
    
    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(p1[:2] - p2[:2])
    
    def calculate_relative_vector(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Calculate relative vector from p1 to p2"""
        return p2[:2] - p1[:2]
    
    def extract_joint_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Extract all joint angles"""
        angles = {}
        
        for joint_name, (pair1, pair2) in self.segmentation.joint_pairs.items():
            try:
                p1, p2 = pair1
                p3, p4 = pair2
                
                if all(i < len(keypoints) for i in [p1, p2, p3, p4]):
                    # p2 and p3 should be the same (joint point)
                    if p2 == p3:
                        angle = self.calculate_angle(
                            keypoints[p1], keypoints[p2], keypoints[p4]
                        )
                        angles[joint_name] = angle
            except:
                angles[joint_name] = 0.0  # Default for failed calculations
                
        return angles
    
    def extract_limb_proportions(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Extract body part proportions and relationships"""
        proportions = {}
        
        try:
            # Torso dimensions
            if all(i < len(keypoints) for i in [11, 12, 23, 24]):
                shoulder_width = self.calculate_distance(keypoints[11], keypoints[12])
                hip_width = self.calculate_distance(keypoints[23], keypoints[24])
                torso_length = self.calculate_distance(
                    (keypoints[11] + keypoints[12]) / 2,  # Average shoulders
                    (keypoints[23] + keypoints[24]) / 2   # Average hips
                )
                
                proportions['shoulder_width'] = shoulder_width
                proportions['hip_width'] = hip_width
                proportions['torso_length'] = torso_length
                proportions['shoulder_hip_ratio'] = shoulder_width / (hip_width + 1e-6)
            
            # Arm lengths
            if all(i < len(keypoints) for i in [11, 13, 15]):
                left_upper_arm = self.calculate_distance(keypoints[11], keypoints[13])
                left_lower_arm = self.calculate_distance(keypoints[13], keypoints[15])
                proportions['left_arm_ratio'] = left_upper_arm / (left_lower_arm + 1e-6)
            
            if all(i < len(keypoints) for i in [12, 14, 16]):
                right_upper_arm = self.calculate_distance(keypoints[12], keypoints[14])
                right_lower_arm = self.calculate_distance(keypoints[14], keypoints[16])
                proportions['right_arm_ratio'] = right_upper_arm / (right_lower_arm + 1e-6)
            
            # Leg lengths
            if all(i < len(keypoints) for i in [23, 25, 27]):
                left_upper_leg = self.calculate_distance(keypoints[23], keypoints[25])
                left_lower_leg = self.calculate_distance(keypoints[25], keypoints[27])
                proportions['left_leg_ratio'] = left_upper_leg / (left_lower_leg + 1e-6)
                
            if all(i < len(keypoints) for i in [24, 26, 28]):
                right_upper_leg = self.calculate_distance(keypoints[24], keypoints[26])
                right_lower_leg = self.calculate_distance(keypoints[26], keypoints[28])
                proportions['right_leg_ratio'] = right_upper_leg / (right_lower_leg + 1e-6)
                
        except Exception as e:
            print(f"Error calculating proportions: {e}")
            
        return proportions
    
    def extract_spatial_relationships(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Extract spatial relationships between body parts"""
        relationships = {}
        
        try:
            # Center of mass approximations
            if all(i < len(keypoints) for i in [11, 12, 23, 24]):
                torso_center = np.mean([keypoints[11], keypoints[12], keypoints[23], keypoints[24]], axis=0)
                
                # Head to torso distance
                if 0 < len(keypoints):
                    head_center = np.mean([keypoints[0], keypoints[1]], axis=0) if len(keypoints) > 1 else keypoints[0]
                    relationships['head_torso_distance'] = self.calculate_distance(head_center, torso_center)
                
                # Limb extensions
                if 15 < len(keypoints):  # Left wrist
                    relationships['left_arm_extension'] = self.calculate_distance(torso_center, keypoints[15])
                if 16 < len(keypoints):  # Right wrist  
                    relationships['right_arm_extension'] = self.calculate_distance(torso_center, keypoints[16])
                if 27 < len(keypoints):  # Left ankle
                    relationships['left_leg_extension'] = self.calculate_distance(torso_center, keypoints[27])
                if 28 < len(keypoints):  # Right ankle
                    relationships['right_leg_extension'] = self.calculate_distance(torso_center, keypoints[28])
                    
        except Exception as e:
            print(f"Error calculating spatial relationships: {e}")
            
        return relationships
