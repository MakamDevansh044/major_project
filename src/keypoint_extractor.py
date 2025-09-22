# File: src/keypoint_extractor.py
import mediapipe as mp
import cv2
import numpy as np
from typing import Optional, List, Tuple

class MediaPipeExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_keypoints(self, image_path: str) -> Optional[np.ndarray]:
        """Extract 33 MediaPipe keypoints from image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([
                    landmark.x, 
                    landmark.y, 
                    landmark.z,
                    landmark.visibility
                ])
            return np.array(landmarks)
        return None
    
    def extract_keypoints_from_video(self, video_path: str) -> List[np.ndarray]:
        """Extract keypoints from video frames"""
        cap = cv2.VideoCapture(video_path)
        keypoints_sequence = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                keypoints_sequence.append(np.array(landmarks))
        
        cap.release()
        return keypoints_sequence
