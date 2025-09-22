# File: src/hierarchical_segmentation.py
import numpy as np
from typing import Dict, List

class HierarchicalBodySegmentation:
    def __init__(self):
        # MediaPipe 33-point landmark indices for body parts
        self.body_parts = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Face landmarks
            'torso': [11, 12, 23, 24],  # Shoulders and hips
            'left_arm': [11, 13, 15, 17, 19, 21],  # Left shoulder to fingers
            'right_arm': [12, 14, 16, 18, 20, 22],  # Right shoulder to fingers  
            'left_leg': [23, 25, 27, 29, 31],  # Left hip to foot
            'right_leg': [24, 26, 28, 30, 32]   # Right hip to foot
        }
        
        # Joint pairs for angle calculation
        self.joint_pairs = {
            'left_elbow': ([11, 13], [13, 15]),    # Shoulder-Elbow, Elbow-Wrist
            'right_elbow': ([12, 14], [14, 16]),
            'left_knee': ([23, 25], [25, 27]),     # Hip-Knee, Knee-Ankle
            'right_knee': ([24, 26], [26, 28]),
            'left_shoulder': ([23, 11], [11, 13]), # Hip-Shoulder, Shoulder-Elbow
            'right_shoulder': ([24, 12], [12, 14]),
            'left_hip': ([11, 23], [23, 25]),      # Shoulder-Hip, Hip-Knee
            'right_hip': ([12, 24], [24, 26])
        }
    
    def segment_body_parts(self, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment keypoints into hierarchical body parts"""
        segments = {}
        for part_name, indices in self.body_parts.items():
            # Filter out invalid indices
            valid_indices = [i for i in indices if i < len(keypoints)]
            segments[part_name] = keypoints[valid_indices]
        return segments
    
    def get_joint_connections(self) -> Dict[str, List[tuple[int, int]]]:
        """Define anatomical connections between joints"""
        connections = {
            'torso_connections': [(11, 12), (11, 23), (12, 24), (23, 24)],
            'arm_connections': [(11, 13), (13, 15), (12, 14), (14, 16)],
            'leg_connections': [(23, 25), (25, 27), (24, 26), (26, 28)]
        }
        return connections
