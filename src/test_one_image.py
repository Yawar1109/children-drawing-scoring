# ============================================================
# test_one_image.py
# Purpose: Load a single image, preprocess it, run it through
#          the trained model for its shape, and print the
#          predicted score.
# Usage:   python src\test_one_image.py "path\to\image.png" "ShapeName"
# Example: python src\test_one_image.py "img1137-D-5.png" "Diamond"
# ============================================================

import os
import sys
import numpy as np
import cv2
import joblib

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
MODELS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "models")

FEATURE_COLS = [
    "ink_ratio", "contour_area", "perimeter", "circularity",
    "solidity", "extent", "aspect_ratio",
    "hu_moment_1", "hu_moment_2", "hu_moment_3", "hu_moment_4",
    "hu_moment_5", "hu_moment_6", "hu_moment_7",
    "num_contours", "paper_brightness", "ink_darkness",
]

SHAPES = ["Circle", "Diamond", "Overlapped Circles",
          "Overlapped Pencils", "Square", "Star", "Triangle", "Wave"]

# ------------------------------------------------------------
# Feature extraction (same pipeline as training)
# ------------------------------------------------------------
import math
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: cannot load image at {image_path}")
        return None
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img[12:-12, 12:-12]

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    ink_mask = binary > 0
    ink_ratio = float(ink_mask.mean())
    paper_pixels = img[~ink_mask]
    paper_brightness = float(paper_pixels.mean()) if paper_pixels.size > 0 else 0.0
    ink_pixels = img[ink_mask]
    ink_darkness = float(255 - ink_pixels.mean()) if ink_pixels.size > 0 else 0.0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    if num_contours == 0:
        return np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    x, y, w, h = cv2.boundingRect(largest)
    bbox = w * h
    extent = area / bbox if bbox > 0 else 0.0
    aspect_ratio = w / h if h > 0 else 0.0
    hu = cv2.HuMoments(cv2.moments(largest)).flatten()
    hu_log = [float(-math.copysign(1.0, v) * math.log10(abs(v) + 1e-30)) for v in hu]

    features = [
        ink_ratio, area, perimeter, circularity, solidity, extent, aspect_ratio,
        hu_log[0], hu_log[1], hu_log[2], hu_log[3], hu_log[4], hu_log[5], hu_log[6],
        num_contours, paper_brightness, ink_darkness,
    ]
    return np.array([features], dtype=np.float32)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src\\test_one_image.py <image_path> <ShapeName>")
        print(f"  ShapeName must be one of: {SHAPES}")
        sys.exit(1)

    image_path = sys.argv[1]
    shape_name = sys.argv[2]

    if shape_name not in SHAPES:
        print(f"ERROR: '{shape_name}' is not a valid shape.")
        print(f"  Valid: {SHAPES}")
        sys.exit(1)

    model_path = os.path.join(MODELS_DIR, f"rf_{shape_name.replace(' ', '_')}.joblib")
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model for {shape_name}...")
    model = joblib.load(model_path)

    print(f"Extracting features from: {image_path}")
    feats = extract_features(image_path)
    if feats is None:
        sys.exit(1)

    print(f"Predicting score...")
    pred = float(model.predict(feats)[0])
    pred_round = int(round(pred))

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print(f"  Image:            {os.path.basename(image_path)}")
    print(f"  Shape folder:     {shape_name}")
    print(f"  Predicted score:  {pred_round}  (raw: {pred:.2f})")
    print("=" * 60)