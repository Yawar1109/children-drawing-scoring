# ============================================================
# check_distribution.py
# Purpose: Compute clean features for all 4,299 images using
#          noise reduction, then print the statistics per shape.
# ============================================================

import os
import math
import numpy as np
import cv2
import pandas as pd

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
METADATA_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")

def measure_image(image_path):
    # 1. Read grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 2. Resize to 128x128
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    
    # 3. Crop border
    img = img[12:-12, 12:-12]
    
    # 4. Noise reduction: Blur the image so small dust dots disappear
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 5. Threshold: Dark pixels become ink (255), background becomes 0
    _, binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 6. Morphological Closing: Fill small gaps in the pencil line
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 7. Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {"circularity": 0.0, "solidity": 0.0, "extent": 0.0, "num_contours": 0}
        
    # 8. Pick largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    
    if perimeter == 0 or area == 0:
        return {"circularity": 0.0, "solidity": 0.0, "extent": 0.0, "num_contours": len(contours)}
        
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    x, y, w, h = cv2.boundingRect(largest)
    bbox_area = w * h
    extent = area / bbox_area if bbox_area > 0 else 0.0
    
    return {
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "num_contours": len(contours)
    }

if __name__ == "__main__":
    print("Loading metadata...")
    df = pd.read_csv(METADATA_CSV).dropna(subset=['score'])
    
    print(f"Measuring {len(df)} images with noise reduction... (approx 2 minutes)")
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 1000 == 0: print(f"  Processed {i}...")
        
        feats = measure_image(row['image_path'])
        if feats:
            feats['shape'] = row['logical_shape']
            rows.append(feats)
            
    out_df = pd.DataFrame(rows)
    
    print("\n" + "="*60)
    print("STATISTICS PER SHAPE (Clean Measurement)")
    print("="*60)
    # This prints the median, min, max, and quartiles for each shape
    stats = out_df.groupby('shape')[['circularity', 'solidity', 'extent', 'num_contours']].describe().round(3)
    
    # Print a simplified version (Median, Min, Max)
    for shape in stats.index:
        print(f"\n--- {shape} ---")
        print(stats.loc[shape, ['circularity', 'solidity', 'num_contours']].to_string())
    
    print("\n" + "="*60)
    print("Done.")