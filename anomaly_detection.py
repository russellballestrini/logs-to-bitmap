#!/usr/bin/env python3
"""
Anomaly Detection Pipeline for HTTP Request Bitmaps

Based on PyImageSearch tutorial: Intro to Anomaly Detection with OpenCV and scikit-learn
Analyzes bitmap images of HTTP requests to identify unusual patterns or anomalous requests.
"""

import cv2
import numpy as np
import os
import glob
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
from pathlib import Path


class RequestBitmapAnomalyDetector:
    def __init__(self, contamination=0.01, n_estimators=100, random_state=42):
        """
        Initialize anomaly detector for HTTP request bitmaps
        
        Args:
            contamination: Expected proportion of outliers (default: 1%)
            n_estimators: Number of isolation trees
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, image_path):
        """
        Extract features from bitmap image using color histograms and texture
        
        Args:
            image_path: Path to the bitmap image
            
        Returns:
            Feature vector as numpy array
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract color histogram features
        # HSV bins: 8 hue, 4 saturation, 4 value
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        
        # Normalize histogram
        cv2.normalize(hist, hist)
        
        # Extract additional features specific to text bitmaps
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Text density (ratio of non-white pixels)
        text_density = np.sum(gray < 250) / gray.size
        
        # Horizontal projection (text line distribution)
        horizontal_projection = np.sum(gray < 250, axis=1)
        horizontal_variance = np.var(horizontal_projection)
        horizontal_mean = np.mean(horizontal_projection)
        
        # Vertical projection (character distribution)  
        vertical_projection = np.sum(gray < 250, axis=0)
        vertical_variance = np.var(vertical_projection)
        vertical_mean = np.mean(vertical_projection)
        
        # Image dimensions
        height, width = gray.shape
        aspect_ratio = width / height
        
        # Combine all features
        features = np.concatenate([
            hist,  # Color histogram
            [text_density, horizontal_variance, horizontal_mean, 
             vertical_variance, vertical_mean, aspect_ratio, height, width]
        ])
        
        return features
    
    def load_dataset(self, bitmap_dir):
        """
        Load all bitmap images and extract features
        
        Args:
            bitmap_dir: Directory containing bitmap files
            
        Returns:
            features: Feature matrix
            image_paths: List of image paths
        """
        bitmap_dir = Path(bitmap_dir)
        
        # Find all BMP files
        image_paths = list(bitmap_dir.glob('*.bmp'))
        image_paths.sort()
        
        if not image_paths:
            raise ValueError(f"No BMP files found in {bitmap_dir}")
        
        print(f"Loading {len(image_paths)} bitmap images...")
        
        features = []
        valid_paths = []
        
        for image_path in image_paths:
            try:
                feature_vector = self.extract_features(image_path)
                features.append(feature_vector)
                valid_paths.append(image_path)
            except Exception as e:
                print(f"Warning: Could not process {image_path}: {e}")
                continue
        
        if not features:
            raise ValueError("No valid features extracted from images")
        
        return np.array(features), valid_paths
    
    def train(self, bitmap_dir):
        """
        Train anomaly detection model on bitmap dataset
        
        Args:
            bitmap_dir: Directory containing training bitmap images
        """
        print("Training anomaly detection model...")
        
        # Load and extract features
        features, image_paths = self.load_dataset(bitmap_dir)
        
        print(f"Extracted features from {len(image_paths)} images")
        print(f"Feature vector size: {features.shape[1]}")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train isolation forest
        self.isolation_forest.fit(features_scaled)
        self.is_trained = True
        
        print("Model training completed!")
        
        # Show training summary
        predictions = self.isolation_forest.predict(features_scaled)
        n_anomalies = np.sum(predictions == -1)
        
        print(f"Training summary:")
        print(f"  Total samples: {len(predictions)}")
        print(f"  Normal samples: {len(predictions) - n_anomalies}")
        print(f"  Anomalies detected: {n_anomalies}")
        print(f"  Anomaly rate: {n_anomalies/len(predictions)*100:.2f}%")
    
    def predict(self, bitmap_path):
        """
        Predict if a single bitmap is anomalous
        
        Args:
            bitmap_path: Path to bitmap image
            
        Returns:
            prediction: 1 for normal, -1 for anomaly
            score: Anomaly score (lower = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        features = self.extract_features(bitmap_path)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.isolation_forest.predict(features_scaled)[0]
        score = self.isolation_forest.score_samples(features_scaled)[0]
        
        return prediction, score
    
    def analyze_dataset(self, bitmap_dir, output_file=None):
        """
        Analyze entire dataset and identify anomalies
        
        Args:
            bitmap_dir: Directory containing bitmap images
            output_file: Optional file to save results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analysis")
        
        # Load dataset
        features, image_paths = self.load_dataset(bitmap_dir)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict all samples
        predictions = self.isolation_forest.predict(features_scaled)
        scores = self.isolation_forest.score_samples(features_scaled)
        
        # Create results
        results = []
        for i, (path, pred, score) in enumerate(zip(image_paths, predictions, scores)):
            results.append({
                'image': path.name,
                'prediction': 'ANOMALY' if pred == -1 else 'NORMAL',
                'score': score,
                'is_anomaly': pred == -1
            })
        
        # Sort by anomaly score (most anomalous first)
        results.sort(key=lambda x: x['score'])
        
        # Print results
        print(f"\nAnomaly Detection Results:")
        print(f"{'Image':<50} {'Status':<10} {'Score':<10}")
        print("-" * 70)
        
        anomaly_count = 0
        for result in results:
            status_marker = "⚠️ " if result['is_anomaly'] else "✅ "
            print(f"{result['image']:<50} {status_marker}{result['prediction']:<8} {result['score']:<10.4f}")
            if result['is_anomaly']:
                anomaly_count += 1
        
        print(f"\nSummary:")
        print(f"Total images analyzed: {len(results)}")
        print(f"Anomalies detected: {anomaly_count}")
        print(f"Normal requests: {len(results) - anomaly_count}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Image,Status,Score\n")
                for result in results:
                    f.write(f"{result['image']},{result['prediction']},{result['score']:.6f}\n")
            print(f"Results saved to: {output_file}")
        
        return results
    
    def save_model(self, model_path):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path):
        """Load trained model from file"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.contamination = model_data['contamination']
        self.n_estimators = model_data['n_estimators']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        print(f"Model loaded from: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly detection for HTTP request bitmaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train and analyze:
    python3 anomaly_detection.py bitmaps/
  
  Train with custom parameters:
    python3 anomaly_detection.py bitmaps/ --contamination 0.05 --estimators 200
  
  Save results to file:
    python3 anomaly_detection.py bitmaps/ --output anomaly_results.csv
  
  Save model for later use:
    python3 anomaly_detection.py bitmaps/ --save-model http_anomaly_model.pkl
        """
    )
    
    parser.add_argument('bitmap_dir', help='Directory containing bitmap images')
    parser.add_argument('--contamination', type=float, default=0.01, 
                       help='Expected proportion of outliers (default: 0.01)')
    parser.add_argument('--estimators', type=int, default=100,
                       help='Number of isolation trees (default: 100)')
    parser.add_argument('--output', '-o', help='Output CSV file for results')
    parser.add_argument('--save-model', help='Save trained model to file')
    parser.add_argument('--load-model', help='Load pre-trained model from file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RequestBitmapAnomalyDetector(
        contamination=args.contamination,
        n_estimators=args.estimators
    )
    
    try:
        # Load existing model or train new one
        if args.load_model:
            detector.load_model(args.load_model)
        else:
            detector.train(args.bitmap_dir)
            
            # Save model if requested
            if args.save_model:
                detector.save_model(args.save_model)
        
        # Analyze dataset
        results = detector.analyze_dataset(args.bitmap_dir, args.output)
        
        # Show top anomalies
        anomalies = [r for r in results if r['is_anomaly']]
        if anomalies:
            print(f"\nTop 5 Most Anomalous Requests:")
            for i, anomaly in enumerate(anomalies[:5]):
                print(f"{i+1}. {anomaly['image']} (score: {anomaly['score']:.4f})")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())