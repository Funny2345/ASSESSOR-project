#!/usr/bin/env python3
"""
Complete image classification pipeline
Usage: python image_classifier.py test.png

Author: Ruichao Mao
Created: 2025-06-13
Last Modified: 2025-06-18
"""

import cv2
import numpy as np
import os
import pandas as pd
from skimage.color import rgb2lab, deltaE_ciede2000
import pickle
from sklearn.preprocessing import StandardScaler
import argparse

class ImageClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        """Load trained model"""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.use_scaled = model_data['use_scaled']
        self.model_name = model_data['model_name']
        self.accuracy = model_data['accuracy']

    def detect_circles(self, image_path):
        """Find circles in image"""
        image = cv2.imread(image_path)
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1.1,
            minDist=30,
            param1=100,
            param2=18,
            minRadius=1,
            maxRadius=80
        )
        
        if circles is None:
            raise ValueError("No circles found")
        
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda x: x[0])
        
        return orig, circles

    def is_valid_circle(self, image, x, y, r):
        """Check if circle is valid"""
        mask = np.zeros(image.shape[:2], dtype="uint8")
        small_radius = max(int(r * 0.5), 1)
        cv2.circle(mask, (x, y), small_radius, 255, -1)
        
        mean_bgr = cv2.mean(image, mask=mask)[:3]
        mean_bgr = np.array([[mean_bgr]], dtype=np.uint8)
        mean_hsv = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV)[0][0]

        min_saturation = 100
        min_value = 100
        max_value = 300
        
        return (mean_hsv[1] >= min_saturation and 
                min_value <= mean_hsv[2] <= max_value)

    def analyze_circles(self, image, circles):
        """Analyze circle colors"""
        results = []
        circle_info = []
        valid_circles = []
        
        for i, (x, y, r) in enumerate(circles):
            if not self.is_valid_circle(image, x, y, r):
                print(f"Circle {i+1} invalid - skipped")
                continue
                
            valid_circles.append((x, y, r))
            
            mask = np.zeros(image.shape[:2], dtype="uint8")
            small_radius = max(int(r * 0.5), 1)
            cv2.circle(mask, (x, y), small_radius, 255, -1)
            
            mean_rgb = cv2.mean(image, mask=mask)[:3]
            rgb_patch = np.array([[mean_rgb[::-1]]], dtype=np.uint8)
            lab_color = rgb2lab(rgb_patch)[0][0]
            
            results.append({
                'Circle': i + 1,
                'R': mean_rgb[2],
                'G': mean_rgb[1],
                'B': mean_rgb[0],
                'L': lab_color[0],
                'a': lab_color[1],
                'b': lab_color[2]
            })
            circle_info.append((x, y, r, small_radius))
        
        return pd.DataFrame(results), circle_info, valid_circles

    def calculate_color_differences(self, df):
        """Calculate color differences"""
        if len(df) < 2:
            return pd.DataFrame()
        
        bg_rgb = np.array([df.iloc[0]['R'], df.iloc[0]['G'], df.iloc[0]['B']])
        bg_lab = np.array([df.iloc[0]['L'], df.iloc[0]['a'], df.iloc[0]['b']])
        
        metrics = []
        for i in range(1, 4):  # Compare first 3 circles
            row = df.iloc[i]
            rgb = np.array([row['R'], row['G'], row['B']])
            lab = np.array([row['L'], row['a'], row['b']])
            
            rgb_distance = np.linalg.norm(rgb - bg_rgb)
            lab_distance = np.linalg.norm(lab - bg_lab)
            de00 = deltaE_ciede2000(bg_lab.reshape(1, 3), lab.reshape(1, 3))[0]
            
            metrics.append({
                'Circle': i + 1,
                'RGB_Distance': rgb_distance,
                'Lab_Distance': lab_distance,
                'CIEDE2000': de00
            })
        
        return pd.DataFrame(metrics)

    def extract_features(self, diff_df):
        """Extract features from color differences"""
        rgb_data = []
        lab_data = [] 
        ciede_data = []

        rgb_data.append(diff_df['RGB_Distance'].tolist())
        lab_data.append(diff_df['Lab_Distance'].tolist())
        ciede_data.append(diff_df['CIEDE2000'].tolist())

        return {
            'rgb': rgb_data,
            'lab': lab_data,
            'ciede': ciede_data
        }

    def predict(self, features):
        """Make predictions using model"""
        test_data = np.array(features['lab'])
        
        if self.use_scaled:
            test_data = self.scaler.transform(test_data)
        
        return self.model.predict(test_data)

    def visualize_results(self, image, circles, circle_info, output_path):
        """Save visualization image"""
        vis = image.copy()
        for i, (x, y, r, small_r) in enumerate(circle_info):
            cv2.circle(vis, (x, y), r, (0, 255, 255), 2)
            cv2.circle(vis, (x, y), small_r, (255, 0, 255), 1)
            cv2.putText(vis, str(i + 1), (x - 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(output_path, vis)

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('image_path', type=str, help='Input image path')
    parser.add_argument('--model', type=str, default='/home/mrc3/project2/co-project/fanyuan/add_6fig_class10/3_ML/saved_models/specified_model_LogisticRegression.pkl', 
                       help='Model file path (default: model.pkl)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]

    print("="*40)
    print(f"Processing image: {args.image_path}")
    print("="*40)

    try:
        # Initialize classifier
        classifier = ImageClassifier(args.model)

        # 1. Detect circles
        print("Step 1: Detecting circles...")
        image, circles = classifier.detect_circles(args.image_path)
        print(f"Found {len(circles)} circles")

        # 2. Analyze colors
        print("Step 2: Analyzing colors...")
        color_data, circle_info, valid_circles = classifier.analyze_circles(image, circles)
        print(f"Found {len(valid_circles)} valid circles")

        if len(valid_circles) == 0:
            raise ValueError("No valid circles found")

        # 3. Calculate color differences
        print("Step 3: Calculating color differences...")
        color_diffs = classifier.calculate_color_differences(color_data)

        # Save intermediate results
        color_data.to_csv(os.path.join(args.output, f"{base_name}_colors.csv"), index=False)
        color_diffs.to_csv(os.path.join(args.output, f"{base_name}_differences.csv"), index=False)

        # 4. Extract features
        print("Step 4: Extracting features...")
        features = classifier.extract_features(color_diffs)

        # 5. Make predictions
        print("Step 5: Making predictions...")
        predictions = classifier.predict(features)

        # 6. Visualize results
        print("Step 6: Visualizing results...")
        classifier.visualize_results(
            image, valid_circles, circle_info, 
            os.path.join(args.output, f"{base_name}_visualization.jpg")
        )

        # Print results
        print("\nClassification Results:")
        print("=" * 40)
        print(f"Model used: {classifier.model_name}")
        print(f"Model accuracy: {classifier.accuracy:.4f}")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1} predicted class: {pred}")
        print("=" * 40)

        print(f"\nDone! Results saved in {args.output}")

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
