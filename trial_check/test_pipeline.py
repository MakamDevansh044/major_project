# File: test_pipeline.py
from src.keypoint_extractor import MediaPipeExtractor
from src.hierarchical_segmentation import HierarchicalBodySegmentation  
from src.geometric_features import GeometricFeatureExtractor
from src.feature_vector_encoder import HierarchicalFeatureEncoder
from src.pose_classifier import HierarchicalPoseClassifier

def test_single_image():
    """Test the complete pipeline on a single image"""
    
    # 1. Extract keypoints
    extractor = MediaPipeExtractor()
    keypoints = extractor.extract_keypoints("image.jpeg")
    
    if keypoints is None:
        print("No pose detected in image")
        return
    
    print(f"✅ Extracted {len(keypoints)} keypoints")
    
    # 2. Create hierarchical features
    encoder = HierarchicalFeatureEncoder()
    features = encoder.create_hierarchical_features(keypoints)
    
    print(f"✅ Created {len(features)} geometric features")
    print("Feature names:", encoder.feature_names[:10])  # First 10 features
    
    # 3. If you have a trained model, load and predict
    # classifier = HierarchicalPoseClassifier()
    # classifier.load_model("models/hierarchical_pose_classifier.pkl")
    # prediction, confidence = classifier.predict(keypoints)
    # print(f"Predicted pose: {prediction} (Confidence: {confidence:.3f})")

if __name__ == "__main__":
    test_single_image()
