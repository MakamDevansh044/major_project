import os
import zipfile
import wget

DATASET_PATH = 'datasets/jhmdb_dataset'
ZIP_URLS = [
    ('http://files.is.tue.mpg.de/jhmdb/JHMDB_video.zip', 'videos'),
    ('http://files.is.tue.mpg.de/jhmdb/splits.zip', 'splits'),
    ('http://files.is.tue.mpg.de/jhmdb/sub_splits.zip', 'sub_splits'),
    ('http://files.is.tue.mpg.de/jhmdb/joint_positions.zip', 'joint_positions')
]

for zip_url, target_dir in ZIP_URLS:
    zip_path = os.path.join(DATASET_PATH, zip_url.split('/')[-1])
    target_dir = os.path.join(DATASET_PATH, target_dir)
    
    if not os.path.exists(target_dir):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        wget.download(zip_url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_PATH)
        