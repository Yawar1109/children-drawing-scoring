# ============================================================
# extract_features_clean.py  (CORRECTED — merges image_path)
# Purpose: Extract 17 features per image on the CLEAN train/test
#          splits using the noise-reduction pipeline.
# ============================================================

import os
import math
import numpy as np
import cv2
import pandas as pd

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
TRAIN_CSV    = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_clean.csv")
TEST_CSV     = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_clean.csv")
METADATA_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")
TRAIN_FEAT   = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_features_clean.csv")
TEST_FEAT    = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_features_clean.csv")

FEATURE_COLS = [
    "ink_ratio",
    "contour_area", "perimeter", "circularity",
    "solidity", "extent", "aspect_ratio",
    "hu_moment_1", "hu_moment_2", "hu_moment_3",
    "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7",
    "num_contours",
    "paper_brightness", "ink_darkness",
]

# ------------------------------------------------------------
# Feature extraction for one image
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
    paper_brightness = float(paper_pixels.mean()) if paper_pixels.size > 0 else float("nan")
    ink_pixels = img[ink_mask]
    ink_darkness = float(255 - ink_pixels.mean()) if ink_pixels.size > 0 else float("nan")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    if num_contours == 0:
        return {
            "ink_ratio": ink_ratio, "contour_area": 0.0, "perimeter": 0.0,
            "circularity": 0.0, "solidity": 0.0, "extent": 0.0, "aspect_ratio": 0.0,
            "hu_moment_1": 0.0, "hu_moment_2": 0.0, "hu_moment_3": 0.0,
            "hu_moment_4": 0.0, "hu_moment_5": 0.0, "hu_moment_6": 0.0,
            "hu_moment_7": 0.0, "num_contours": 0,
            "paper_brightness": paper_brightness, "ink_darkness": ink_darkness,
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

    return {
        "ink_ratio": ink_ratio, "contour_area": area, "perimeter": perimeter,
        "circularity": circularity, "solidity": solidity, "extent": extent,
        "aspect_ratio": aspect_ratio,
        "hu_moment_1": hu_log[0], "hu_moment_2": hu_log[1], "hu_moment_3": hu_log[2],
        "hu_moment_4": hu_log[3], "hu_moment_5": hu_log[4], "hu_moment_6": hu_log[5],
        "hu_moment_7": hu_log[6], "num_contours": num_contours,
        "paper_brightness": paper_brightness, "ink_darkness": ink_darkness,
    }

# ------------------------------------------------------------
# Process a split (merges image_path first)
# ------------------------------------------------------------
def process(meta_csv, out_csv, split_name):
    print(f"\n=== {split_name} ===")
    df = pd.read_csv(meta_csv)
    print(f"Rows: {len(df)}")

    # Merge image_path from dataset_metadata.csv
    meta = pd.read_csv(METADATA_CSV)
    path_lookup = dict(zip(meta["filename"], meta["image_path"]))
    df["image_path"] = df["filename"].map(path_lookup)
    missing = int(df["image_path"].isna().sum())
    if missing > 0:
        print(f"  WARNING: {missing} rows missing image_path — will be skipped")

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 500 == 0 or i == 1:
            print(f"  [{split_name}] {i}/{len(df)}...")
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

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    process(TRAIN_CSV, TRAIN_FEAT, "train")
    process(TEST_CSV, TEST_FEAT, "test")
    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)