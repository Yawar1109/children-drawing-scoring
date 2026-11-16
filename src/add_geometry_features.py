# ============================================================
# add_geometry_features.py
# Purpose: Add 4 more features requested by the professor:
#          closure, symmetry, num_angles, mean_angle.
#          Combines them with the 17 existing features = 21 total.
# Output:  train_features_plus.csv, test_features_plus.csv
# ============================================================

import os
import math
import numpy as np
import cv2
import pandas as pd

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
TRAIN_CLEAN = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_clean.csv")
TEST_CLEAN  = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_clean.csv")
METADATA_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")
TRAIN_PLUS = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_features_plus.csv")
TEST_PLUS  = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_features_plus.csv")

FEATURE_COLS = [
    "ink_ratio", "contour_area", "perimeter", "circularity",
    "solidity", "extent", "aspect_ratio",
    "hu_moment_1", "hu_moment_2", "hu_moment_3", "hu_moment_4",
    "hu_moment_5", "hu_moment_6", "hu_moment_7",
    "num_contours", "paper_brightness", "ink_darkness",
    # NEW 4 features from professor's list:
    "closure", "symmetry", "num_angles", "mean_angle",
]

# ------------------------------------------------------------
# NEW FEATURES
# ------------------------------------------------------------
def compute_closure(contour, perimeter):
    """Return 1 if the contour is closed (no big gap), else 0."""
    if perimeter == 0 or len(contour) < 5:
        return 0
    # Compute gap between first and last point of the contour
    first = contour[0][0]
    last = contour[-1][0]
    gap = float(np.sqrt((first[0] - last[0])**2 + (first[1] - last[1])**2))
    # Consider closed if gap is < 10% of perimeter
    return 1 if gap < 0.10 * perimeter else 0

def compute_symmetry(ink_mask):
    """Compare ink mask with its mirror image. Return 0..1."""
    if ink_mask.sum() < 10:
        return 0.0
    flipped = cv2.flip(ink_mask.astype(np.uint8), 1)
    total_ink = float(ink_mask.sum()) + float(flipped.sum())
    overlap = float(np.logical_and(ink_mask, flipped.astype(bool)).sum())
    if total_ink == 0:
        return 0.0
    # Dice coefficient = 2 * |A ∩ B| / (|A| + |B|)
    return float(2.0 * overlap / total_ink)

def compute_angles(contour):
    """Return (num_angles, mean_angle_degrees)."""
    if len(contour) < 5:
        return 0, 0.0
    # Approx polygon: epsilon proportional to perimeter
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    n = len(approx)
    if n < 3:
        return 0, 0.0

    # Compute interior angles
    pts = approx.reshape(-1, 2)
    angles = []
    for i in range(n):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v1 = p0 - p1
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        angles.append(angle_deg)

    if not angles:
        return n, 0.0
    return n, float(np.mean(angles))

# ------------------------------------------------------------
# Full extraction (21 features)
# ------------------------------------------------------------
def extract_one(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
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
        return {
            **{c: 0.0 for c in FEATURE_COLS[:14]},
            "num_contours": 0, "paper_brightness": paper_brightness,
            "ink_darkness": ink_darkness,
            "closure": 0, "symmetry": 0.0, "num_angles": 0, "mean_angle": 0.0,
        }

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

    # NEW FEATURES
    closure_val = compute_closure(largest, perimeter)
    symmetry_val = compute_symmetry(ink_mask)
    num_angles, mean_angle = compute_angles(largest)

    return {
        "ink_ratio": ink_ratio, "contour_area": area, "perimeter": perimeter,
        "circularity": circularity, "solidity": solidity, "extent": extent,
        "aspect_ratio": aspect_ratio,
        "hu_moment_1": hu_log[0], "hu_moment_2": hu_log[1], "hu_moment_3": hu_log[2],
        "hu_moment_4": hu_log[3], "hu_moment_5": hu_log[4], "hu_moment_6": hu_log[5],
        "hu_moment_7": hu_log[6],
        "num_contours": num_contours,
        "paper_brightness": paper_brightness, "ink_darkness": ink_darkness,
        "closure": closure_val, "symmetry": symmetry_val,
        "num_angles": num_angles, "mean_angle": mean_angle,
    }

# ------------------------------------------------------------
# Process
# ------------------------------------------------------------
def process(clean_csv, out_csv, name):
    print(f"\n=== {name} ===")
    df = pd.read_csv(clean_csv)
    meta = pd.read_csv(METADATA_CSV)
    path_lookup = dict(zip(meta["filename"], meta["image_path"]))
    df["image_path"] = df["filename"].map(path_lookup)

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 500 == 0 or i == 1:
            print(f"  [{name}] {i}/{len(df)}...")
        if pd.isna(row["image_path"]):
            continue
        feats = extract_one(row["image_path"])
        if feats is None:
            continue
        feats["filename"] = row["filename"]
        feats["logical_shape"] = row["logical_shape"]
        feats["score"] = int(row["score"])
        rows.append(feats)

    out = pd.DataFrame(rows, columns=["filename", "logical_shape", "score"] + FEATURE_COLS)
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Shape: {out.shape}")

if __name__ == "__main__":
    process(TRAIN_CLEAN, TRAIN_PLUS, "train")
    process(TEST_CLEAN, TEST_PLUS, "test")
    print("\n" + "=" * 60)
    print("DONE — 21 features extracted")
    print("=" * 60)