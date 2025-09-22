# File: src/pose_classifier.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
from typing import Tuple, Dict, List
from feature_vector_encoder import HierarchicalFeatureEncoder

class HierarchicalPoseClassifier:
    def __init__(self, classifier_type='random_forest'):
        self.classifier_type = classifier_type
        self.pipeline = None
        self.feature_encoder = HierarchicalFeatureEncoder()
        self.class_labels = []
        
    def _create_classifier(self):
        """Create classifier pipeline"""
        if self.classifier_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.classifier_type == 'svm':
            classifier = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        # Create pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
    
    def prepare_dataset(self, keypoints_data: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix from keypoints data"""
        features_list = []
        valid_labels = []
        
        for keypoints, label in zip(keypoints_data, labels):
            if keypoints is not None and len(keypoints) > 0:
                features = self.feature_encoder.create_hierarchical_features(keypoints)
                if len(features) > 0:
                    features_list.append(features)
                    valid_labels.append(label)
        
        if not features_list:
            raise ValueError("No valid features extracted from keypoints data")
        
        # Ensure all feature vectors have the same length
        min_length = min(len(f) for f in features_list)
        features_matrix = np.array([f[:min_length] for f in features_list])
        
        return features_matrix, np.array(valid_labels)
    
    def train(self, keypoints_data: List[np.ndarray], labels: List[str]) -> Dict:
        """Train the pose classifier"""
        # Prepare dataset
        X, y = self.prepare_dataset(keypoints_data, labels)
        self.class_labels = list(set(labels))
        
        # Create classifier
        self._create_classifier()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        
        # Cross validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5)
        
        # Predictions for detailed analysis
        y_pred = self.pipeline.predict(X_test)
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_names': self.feature_encoder.feature_names
        }
        
        return results
    
    def predict(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """Predict pose class for single keypoint set"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        features = self.feature_encoder.create_hierarchical_features(keypoints)
        if len(features) == 0:
            return "unknown", 0.0
        
        features = features.reshape(1, -1)
        prediction = self.pipeline.predict(features)[0]
        
        # Get prediction probability if available
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            probabilities = self.pipeline.predict_proba(features)[0]
            max_prob = max(probabilities)
        else:
            max_prob = 1.0
        
        return prediction, max_prob
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for interpretability"""
        if self.pipeline is None or self.classifier_type != 'random_forest':
            return {}
        
        rf_classifier = self.pipeline.named_steps['classifier']
        importance_dict = {}
        
        for i, importance in enumerate(rf_classifier.feature_importances_):
            if i < len(self.feature_encoder.feature_names):
                feature_name = self.feature_encoder.feature_names[i]
                importance_dict[feature_name] = importance
                
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'pipeline': self.pipeline,
            'feature_encoder': self.feature_encoder,
            'class_labels': self.class_labels,
            'classifier_type': self.classifier_type
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.feature_encoder = model_data['feature_encoder']
        self.class_labels = model_data['class_labels']
        self.classifier_type = model_data['classifier_type']
