# Create this test file: quick_test.py
from src.keypoint_extractor import MediaPipeExtractor
import cv2

def test_image():
    extractor = MediaPipeExtractor()
    
    # Test with your uploaded image
    keypoints = extractor.extract_keypoints("image.jpeg")
    
    if keypoints is not None:
        print(f"✅ SUCCESS: Extracted {len(keypoints)} keypoints")
        print(f"First keypoint: {keypoints[0]}")
        return True
    else:
        print("❌ FAILED: No pose detected")
        return False

if __name__ == "__main__":
    test_image()
