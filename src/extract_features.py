# ============================================================
# extract_features.py
# Purpose: extract 17 features from each processed image
#          (15 standard + 2 confound features) and save to CSV.
# Inputs:  outputs/tables/train_metadata.csv
#          outputs/tables/test_metadata.csv
#          outputs/processed_images/   (128x128 grayscale PNGs)
# Outputs: outputs/tables/train_features.csv
#          outputs/tables/test_features.csv
# Author:  Yawar Sadiq  (mentored step-by-step)
# ============================================================

import os
import math
import numpy as np
import cv2
import pandas as pd

# ------------------------------------------------------------
# 1. Paths (do not change unless files moved)
# ------------------------------------------------------------
PROJECT_ROOT     = r"C:\Users\dell\Documents\ShapeScoringProject"
PROCESSED_DIR    = os.path.join(PROJECT_ROOT, "outputs", "processed_images")
TRAIN_META_CSV   = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_metadata.csv")
TEST_META_CSV    = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_metadata.csv")
TRAIN_FEAT_CSV   = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_features.csv")
TEST_FEAT_CSV    = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_features.csv")

# ------------------------------------------------------------
# 2. Constants
# ------------------------------------------------------------
BORDER_MARGIN = 12       # pixels to remove from each side (128 -> 104 usable)
FEATURE_COLUMNS = [
    "ink_ratio",
    "contour_area", "perimeter", "circularity",
    "solidity", "extent", "aspect_ratio",
    "hu_moment_1", "hu_moment_2", "hu_moment_3",
    "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7",
    "num_contours",
    "paper_brightness", "ink_darkness",
]

# ------------------------------------------------------------
# 3. Feature extraction for one image
# ------------------------------------------------------------
def extract_one_image(image_path):
    """
    Read one processed image, crop border, and return a dict of
    17 features. Return None if the image cannot be read.
    """
    # Read as grayscale (our processed images are grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Crop away the black square border
    h, w = img.shape
    if h <= 2 * BORDER_MARGIN or w <= 2 * BORDER_MARGIN:
        # Image too small to crop safely
        return None
    img = img[BORDER_MARGIN:h - BORDER_MARGIN, BORDER_MARGIN:w - BORDER_MARGIN]

    # --- 3.1 Basic ink/paper masks ----------------------------
    # Otsu threshold: finds the "best" cutoff between ink and paper
    # for THIS image (adaptive to brightness).
    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # After Otsu, ink is 0 (black), paper is 255 (white)
    ink_mask = (binary == 0)

    # ink_ratio = fraction of dark pixels
    ink_ratio = float(ink_mask.mean())

    # --- 3.2 Two confound features ----------------------------
    # paper_brightness = mean brightness of paper pixels
    paper_pixels = img[~ink_mask]
    paper_brightness = float(paper_pixels.mean()) if paper_pixels.size > 0 else float("nan")

    # ink_darkness = 255 - mean brightness of ink pixels
    ink_pixels = img[ink_mask]
    ink_darkness = float(255 - ink_pixels.mean()) if ink_pixels.size > 0 else float("nan")

    # --- 3.3 Find contours (the drawing outlines) ------------
    # We invert the binary because findContours expects the
    # foreground (the drawing) to be white on a black background.
    ink_for_contours = (~ink_mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink_for_contours,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    num_contours = len(contours)

    if num_contours == 0:
        # No ink detected at all — return zeros for shape features
        return {
            "ink_ratio": ink_ratio,
            "contour_area": 0.0, "perimeter": 0.0, "circularity": 0.0,
            "solidity": 0.0, "extent": 0.0, "aspect_ratio": 0.0,
            "hu_moment_1": 0.0, "hu_moment_2": 0.0, "hu_moment_3": 0.0,
            "hu_moment_4": 0.0, "hu_moment_5": 0.0, "hu_moment_6": 0.0,
            "hu_moment_7": 0.0,
            "num_contours": 0,
            "paper_brightness": paper_brightness,
            "ink_darkness": ink_darkness,
        }

    # Pick the LARGEST contour (this is the main drawing)
    largest = max(contours, key=cv2.contourArea)

    # --- 3.4 Shape features from the largest contour ---------
    contour_area = float(cv2.contourArea(largest))
    perimeter    = float(cv2.arcLength(largest, True))

    # circularity = 4*pi*area / perimeter^2   (1.0 = perfect circle)
    if perimeter > 0:
        circularity = float(4 * math.pi * contour_area / (perimeter ** 2))
    else:
        circularity = 0.0

    # solidity = area / convex-hull-area
    hull       = cv2.convexHull(largest)
    hull_area  = float(cv2.contourArea(hull))
    solidity   = float(contour_area / hull_area) if hull_area > 0 else 0.0

    # extent = area / bounding-box-area
    x, y, bw, bh = cv2.boundingRect(largest)
    bbox_area = float(bw * bh)
    extent    = float(contour_area / bbox_area) if bbox_area > 0 else 0.0

    # aspect_ratio = width / height of bounding box
    aspect_ratio = float(bw / bh) if bh > 0 else 0.0

    # --- 3.5 Hu moments (7 shape descriptors) ----------------
    # cv2.HuMoments returns an array of shape (7, 1)
    hu = cv2.HuMoments(cv2.moments(largest)).flatten()
    # Log-transform is standard: values are tiny and span many
    # orders of magnitude; log makes them comparable.
    # We add a tiny epsilon to avoid log(0).
    hu_log = [float(-math.copysign(1.0, v) * math.log10(abs(v) + 1e-30)) for v in hu]

    # --- 3.6 Assemble and return -----------------------------
    return {
        "ink_ratio": ink_ratio,
        "contour_area": contour_area,
        "perimeter": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "aspect_ratio": aspect_ratio,
        "hu_moment_1": hu_log[0],
        "hu_moment_2": hu_log[1],
        "hu_moment_3": hu_log[2],
        "hu_moment_4": hu_log[3],
        "hu_moment_5": hu_log[4],
        "hu_moment_6": hu_log[5],
        "hu_moment_7": hu_log[6],
        "num_contours": num_contours,
        "paper_brightness": paper_brightness,
        "ink_darkness": ink_darkness,
    }

# ------------------------------------------------------------
# 4. Process one metadata CSV -> one features CSV
# ------------------------------------------------------------
def process_metadata(metadata_csv, output_csv, split_name):
    print(f"\n=== Processing {split_name} split ===")
    df = pd.read_csv(metadata_csv)
    print(f"Rows in {split_name} metadata: {len(df)}")

    rows = []
    failed = 0
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i % 500 == 0 or i == 1:
            print(f"  [{split_name}] image {i} of {total}...")

        image_path = os.path.join(PROCESSED_DIR, row["filename"])
        feats = extract_one_image(image_path)

        if feats is None:
            failed += 1
            continue

        # Attach identity and label columns so we can use them later
        feats["filename"]       = row["filename"]
        feats["logical_shape"]  = row["logical_shape"]
        feats["score"]          = int(row["score"])
        rows.append(feats)

    print(f"  [{split_name}] OK: {len(rows)}  Failed: {failed}")

    if len(rows) == 0:
        print(f"ERROR: no features extracted for {split_name}. Stopping.")
        raise SystemExit(1)

    out_df = pd.DataFrame(rows, columns=["filename", "logical_shape", "score"] + FEATURE_COLUMNS)
    out_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    print(f"  Shape: {out_df.shape[0]} rows x {out_df.shape[1]} columns")
    return out_df

# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    train_df = process_metadata(TRAIN_META_CSV, TRAIN_FEAT_CSV, "train")
    test_df  = process_metadata(TEST_META_CSV,  TEST_FEAT_CSV,  "test")

    print("\n" + "=" * 60)
    print("DONE.")
    print(f"train_features.csv: {len(train_df)} rows")
    print(f"test_features.csv : {len(test_df)} rows")
    print("=" * 60)
    print("\nFirst 3 rows of train_features.csv (a few columns):")
    preview_cols = ["filename", "logical_shape", "score",
                    "circularity", "solidity", "num_contours",
                    "paper_brightness", "ink_darkness"]
    print(train_df[preview_cols].head(3).to_string(index=False))