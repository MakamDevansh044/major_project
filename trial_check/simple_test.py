import numpy as np
from src.keypoint_extractor import MediaPipeExtractor
from src.hierarchical_segmentation import HierarchicalBodySegmentation
from src.geometric_features import GeometricFeatureExtractor
from src.feature_vector_encoder import HierarchicalFeatureEncoder

def simple_test():
    print("🧪 Testing Hierarchical Pose Pipeline...")
    
    # Test with your image
    extractor = MediaPipeExtractor()
    keypoints = extractor.extract_keypoints("image.jpeg")
    
    if keypoints is None:
        print("❌ No pose detected in image")
        return
    
    print(f"✅ Step 1: Extracted {len(keypoints)} keypoints")
    
    # Test hierarchical segmentation
    segmentation = HierarchicalBodySegmentation()
    body_parts = segmentation.segment_body_parts(keypoints)
    print(f"✅ Step 2: Segmented into {len(body_parts)} body parts")
    
    # Test geometric features
    feature_extractor = GeometricFeatureExtractor(segmentation)
    angles = feature_extractor.extract_joint_angles(keypoints)
    proportions = feature_extractor.extract_limb_proportions(keypoints)
    
    print(f"✅ Step 3: Extracted {len(angles)} joint angles")
    print(f"✅ Step 4: Extracted {len(proportions)} body proportions")
    
    # Test feature encoding
    encoder = HierarchicalFeatureEncoder()
    features = encoder.create_hierarchical_features(keypoints)
    
    print(f"✅ Step 5: Created feature vector of length {len(features)}")
    print(f"✅ Feature names: {encoder.feature_names[:5]}...")
    
    print("\n🎉 All components working! Ready for full pipeline.")

if __name__ == "__main__":
    simple_test()