# ============================================================
# inspect_shape_folders.py
# Purpose: flag likely wrong-shape images inside each folder.
# Method : compute 3 simple shape numbers per image using a
#          FIXED threshold (not Otsu). Flag using simple rules.
#          No image is modified, moved, or uploaded.
# Inputs : outputs/tables/dataset_metadata.csv
#          Original dataset images (read-only)
# Outputs: outputs/tables/shape_inspection_summary.csv
#          outputs/tables/shape_inspection_flags_PRIVATE.csv
#          outputs/tables/shape_inspection_filenames_PRIVATE.csv
# ============================================================

import os
import math
import numpy as np
import cv2
import pandas as pd

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
PROJECT_ROOT    = r"C:\Users\dell\Documents\ShapeScoringProject"
METADATA_CSV    = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")
SUMMARY_CSV     = os.path.join(PROJECT_ROOT, "outputs", "tables", "shape_inspection_summary.csv")
FLAGS_CSV       = os.path.join(PROJECT_ROOT, "outputs", "tables", "shape_inspection_flags_PRIVATE.csv")
FILENAMES_CSV   = os.path.join(PROJECT_ROOT, "outputs", "tables", "shape_inspection_filenames_PRIVATE.csv")

# ------------------------------------------------------------
# 2. Constants
# ------------------------------------------------------------
FIXED_THRESHOLD = 128       # pixels darker than this = ink
BORDER_MARGIN   = 12        # crop 12 px from each side
# The original dataset images are large and not necessarily
# 128x128. We resize on the fly so measurement is consistent.
RESIZE_TO       = 128

# ------------------------------------------------------------
# 3. Flagging rules (one per logical_shape)
# Each rule is a function that takes a dict of features and
# returns True if the image is SUSPICIOUS.
# ------------------------------------------------------------
def flag_circle(f):
    return f["circularity"] < 0.50 or f["solidity"] < 0.70

def flag_square(f):
    return f["solidity"] < 0.70 or f["extent"] < 0.50

def flag_diamond(f):
    return f["solidity"] < 0.70 or f["extent"] < 0.50

def flag_triangle(f):
    return f["solidity"] < 0.70 or f["circularity"] > 0.80

def flag_star(f):
    return f["solidity"] > 0.85

def flag_wave(f):
    return f["circularity"] > 0.50

def flag_overlapped_circles(f):
    return f["num_contours"] < 2 or f["solidity"] < 0.50

def flag_overlapped_pencils(f):
    return f["num_contours"] < 2 or f["solidity"] < 0.50

FLAGGERS = {
    "Circle":              flag_circle,
    "Square":              flag_square,
    "Diamond":             flag_diamond,
    "Triangle":            flag_triangle,
    "Star":                flag_star,
    "Wave":                flag_wave,
    "Overlapped Circles":  flag_overlapped_circles,
    "Overlapped Pencils":  flag_overlapped_pencils,
}

# ------------------------------------------------------------
# 4. Measure one image
# ------------------------------------------------------------
def measure_one_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to 128x128 for consistency
    img = cv2.resize(img, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)

    # Crop the border
    h, w = img.shape
    if h <= 2 * BORDER_MARGIN or w <= 2 * BORDER_MARGIN:
        return None
    img = img[BORDER_MARGIN:h - BORDER_MARGIN, BORDER_MARGIN:w - BORDER_MARGIN]

    # Fixed threshold: ink = dark
    ink_mask = (img < FIXED_THRESHOLD)
    ink_ratio = float(ink_mask.mean())

    # Find contours
    ink_for_contours = ink_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink_for_contours,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    if num_contours == 0:
        return {
            "circularity": 0.0, "solidity": 0.0, "extent": 0.0,
            "num_contours": 0, "ink_ratio": ink_ratio,
        }

    largest = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(largest))
    perimeter    = float(cv2.arcLength(largest, True))

    circularity = (4 * math.pi * contour_area / (perimeter ** 2)) if perimeter > 0 else 0.0

    hull      = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity  = (contour_area / hull_area) if hull_area > 0 else 0.0

    x, y, bw, bh = cv2.boundingRect(largest)
    bbox_area = float(bw * bh)
    extent    = (contour_area / bbox_area) if bbox_area > 0 else 0.0

    return {
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "num_contours": num_contours,
        "ink_ratio": ink_ratio,
    }

# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Loading metadata...")
    df = pd.read_csv(METADATA_CSV)
    df = df.dropna(subset=["score"])
    df["score"] = df["score"].astype(int)
    print(f"Total images: {len(df)}")

    # --- 5.1 Loop over images ---------------------------------
    all_rows = []
    total = len(df)
    failed = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i % 500 == 0 or i == 1:
            print(f"  Processing {i} of {total}...")

        feats = measure_one_image(row["image_path"])
        if feats is None:
            failed += 1
            continue

        feats["filename"]      = row["filename"]
        feats["logical_shape"] = row["logical_shape"]
        all_rows.append(feats)

    print(f"Done. OK: {len(all_rows)}  Failed: {failed}")

    per_image = pd.DataFrame(all_rows)

    # --- 5.2 Apply flagging rules per shape --------------------
    print("\nApplying flagging rules...")
    suspicion_flags = []
    for _, row in per_image.iterrows():
        shape = row["logical_shape"]
        flagger = FLAGGERS.get(shape)
        if flagger is None:
            suspicion_flags.append(False)
            continue
        suspicion_flags.append(bool(flagger(row.to_dict())))
    per_image["suspicious"] = suspicion_flags

    # --- 5.3 Build summary per folder -------------------------
    summary_rows = []
    for shape, group in per_image.groupby("logical_shape"):
        n_total    = len(group)
        n_susp     = int(group["suspicious"].sum())
        pct        = round(100.0 * n_susp / n_total, 1) if n_total > 0 else 0.0
        summary_rows.append({
            "logical_shape": shape,
            "total_images":  n_total,
            "suspicious_count": n_susp,
            "suspicious_percent": pct,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("logical_shape")

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary to: {SUMMARY_CSV}")
    print(summary_df.to_string(index=False))

    # --- 5.4 Build private list of suspects -------------------
    # Two separate files:
    #  - FLAGS: anonymous IDs + features (safe to share)
    #  - FILENAMES: anonymous ID + real filename (private)
    suspects = per_image[per_image["suspicious"]].copy().reset_index(drop=True)
    suspects["suspect_id"] = ["SUS_%04d" % (i + 1) for i in range(len(suspects))]

    # safe file (anonymous IDs only)
    safe_cols = ["suspect_id", "logical_shape", "circularity",
                 "solidity", "extent", "num_contours"]
    suspects[safe_cols].round(4).to_csv(FLAGS_CSV, index=False)

    # private file (real filenames)
    private_cols = ["suspect_id", "filename"]
    suspects[private_cols].to_csv(FILENAMES_CSV, index=False)

    print(f"\nTotal suspects flagged: {len(suspects)}")
    print(f"Safe summary file (shareable):       {FLAGS_CSV}")
    print(f"Private filenames file (do not share): {FILENAMES_CSV}")