# File: src/main_pipeline.py
import os
import numpy as np
from typing import List, Dict
import glob
from keypoint_extractor import MediaPipeExtractor
from pose_classifier import HierarchicalPoseClassifier

def load_jhmdb_data(dataset_path: str, max_samples_per_class: int = 50) -> tuple[List[np.ndarray], List[str]]:
    """Load JHMDB dataset and extract keypoints"""
    extractor = MediaPipeExtractor()
    keypoints_data = []
    labels = []
    
    # Get all video files
    video_files = glob.glob(os.path.join(dataset_path, "**/*.avi"), recursive=True)
    
    class_counts = {}
    
    for video_path in video_files:
        # Extract class name from path
        class_name = os.path.basename(os.path.dirname(video_path))
        
        # Limit samples per class
        if class_counts.get(class_name, 0) >= max_samples_per_class:
            continue
            
        print(f"Processing: {video_path}")
        
        # Extract keypoints from video
        keypoints_sequence = extractor.extract_keypoints_from_video(video_path)
        
        if keypoints_sequence:
            # Take middle frame or average multiple frames
            mid_frame = len(keypoints_sequence) // 2
            if mid_frame < len(keypoints_sequence):
                keypoints_data.append(keypoints_sequence[mid_frame])
                labels.append(class_name)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return keypoints_data, labels

def main():
    """Main pipeline execution"""
    print("🚀 Starting Hierarchical Pose Classification Pipeline")
    
    # Configuration
    DATASET_PATH = "datasets/jhmdb_dataset"
    MODEL_SAVE_PATH = "models/hierarchical_pose_classifier.pkl"
    
    # 1. Load and prepare data
    print("📁 Loading JHMDB dataset...")
    keypoints_data, labels = load_jhmdb_data(DATASET_PATH, max_samples_per_class=30)
    
    print(f"✅ Loaded {len(keypoints_data)} samples from {len(set(labels))} classes")
    print(f"Classes: {set(labels)}")
    
    # 2. Create and train classifier
    print("🤖 Training hierarchical pose classifier...")
    classifier = HierarchicalPoseClassifier(classifier_type='random_forest')
    
    results = classifier.train(keypoints_data, labels)
    
    # 3. Print results
    print("📊 Training Results:")
    print(f"Training Accuracy: {results['train_accuracy']:.3f}")
    print(f"Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"Cross-Validation: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    # 4. Feature importance analysis
    print("\n🔍 Top Important Features:")
    feature_importance = classifier.get_feature_importance()
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"{i+1}. {feature}: {importance:.3f}")
    
    # 5. Save model
    os.makedirs("models", exist_ok=True)
    classifier.save_model(MODEL_SAVE_PATH)
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    
    # 6. Test single prediction
    print("\n🧪 Testing single prediction...")
    if keypoints_data:
        test_keypoints = keypoints_data[0]
        prediction, confidence = classifier.predict(test_keypoints)
        print(f"Prediction: {prediction} (Confidence: {confidence:.3f})")
    
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
