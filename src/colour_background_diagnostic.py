# ============================================================
# colour_background_diagnostic.py
# Purpose: check whether pen colour / paper brightness is
# accidentally related to the score (a confounding variable).
# Input:  outputs/tables/dataset_metadata.csv  (all 4,299 images)
# Output: outputs/tables/colour_background_diagnostic.csv
# Author: Yawar Sadiq  (mentored step-by-step)
# ============================================================

import os
import csv
import numpy as np
import cv2
import pandas as pd

# ------------------------------------------------------------
# 1. Paths (do not change these unless the files moved)
# ------------------------------------------------------------
PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
METADATA_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "outputs", "tables", "colour_background_diagnostic.csv")

# ------------------------------------------------------------
# 2. Load metadata
# ------------------------------------------------------------
print("Loading metadata...")
df = pd.read_csv(METADATA_CSV)
df = df.dropna(subset=["score"])           # keep rows with a valid score
df["score"] = df["score"].astype(int)
print(f"Total images to analyse: {len(df)}")

# ------------------------------------------------------------
# 3. Helper: compute colour numbers for ONE image
# ------------------------------------------------------------
def measure_one_image(image_path):
    """
    Return a dict of colour / brightness numbers for one image.
    If the image cannot be read, return None.
    """
    # Read in COLOUR (BGR order, that is how OpenCV reads it)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Convert BGR -> RGB so our numbers are in the usual R,G,B order
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Convert to grayscale and to HSV (both used for simple stats)
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv  = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    # Whole-image stats
    mean_r = float(img_rgb[:, :, 0].mean())
    mean_g = float(img_rgb[:, :, 1].mean())
    mean_b = float(img_rgb[:, :, 2].mean())
    mean_s = float(hsv[:, :, 1].mean())   # saturation (how colourful)
    mean_v = float(hsv[:, :, 2].mean())   # brightness

    # Separate "ink" (dark pixels) and "paper" (bright pixels)
    # Using a threshold of 128 on grayscale is a simple, safe rule.
    ink_mask   = gray < 128
    paper_mask = gray >= 128

    if ink_mask.sum() > 0:
        ink_r = float(img_rgb[:, :, 0][ink_mask].mean())
        ink_g = float(img_rgb[:, :, 1][ink_mask].mean())
        ink_b = float(img_rgb[:, :, 2][ink_mask].mean())
    else:
        ink_r = ink_g = ink_b = float("nan")

    if paper_mask.sum() > 0:
        paper_brightness = float(gray[paper_mask].mean())
    else:
        paper_brightness = float("nan")

    ink_ratio = float(ink_mask.mean())

    return {
        "mean_R": mean_r, "mean_G": mean_g, "mean_B": mean_b,
        "mean_saturation": mean_s, "mean_brightness": mean_v,
        "ink_R": ink_r, "ink_G": ink_g, "ink_B": ink_b,
        "paper_brightness": paper_brightness,
        "ink_ratio": ink_ratio,
    }

# ------------------------------------------------------------
# 4. Loop over every image, collect per-image numbers
# ------------------------------------------------------------
rows = []
total = len(df)
failed = 0
for i, (_, row) in enumerate(df.iterrows(), start=1):
    if i % 500 == 0 or i == 1:
        print(f"  Processing image {i} of {total}...")

    result = measure_one_image(row["image_path"])
    if result is None:
        failed += 1
        continue

    # Add identity columns alongside the colour stats
    result["logical_shape"] = row["logical_shape"]
    result["score"] = int(row["score"])
    rows.append(result)

print(f"Finished. OK: {len(rows)}  Failed to read: {failed}")

if len(rows) == 0:
    print("ERROR: no images were readable. Stopping.")
    raise SystemExit(1)

# ------------------------------------------------------------
# 5. Build per-image DataFrame, then aggregate by (shape, score)
# ------------------------------------------------------------
per_image = pd.DataFrame(rows)

# Numeric columns that we will average
numeric_cols = [
    "mean_R", "mean_G", "mean_B",
    "mean_saturation", "mean_brightness",
    "ink_R", "ink_G", "ink_B",
    "paper_brightness", "ink_ratio",
]

grouped = (
    per_image
    .groupby(["logical_shape", "score"])[numeric_cols]
    .mean()
    .reset_index()
)

# Count how many images per (shape, score) group
counts = (
    per_image
    .groupby(["logical_shape", "score"])
    .size()
    .reset_index(name="n_images")
)

summary = grouped.merge(counts, on=["logical_shape", "score"])

# Round for readability
for col in numeric_cols:
    summary[col] = summary[col].round(2)

# Sort by shape then score (nicer to read)
summary = summary.sort_values(["logical_shape", "score"]).reset_index(drop=True)

# ------------------------------------------------------------
# 6. Save the small CSV (numbers only — no images leave the machine)
# ------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
summary.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("DONE.")
print(f"Saved aggregate diagnostic to: {OUTPUT_CSV}")
print(f"Rows in summary: {len(summary)}")
print("=" * 60)
print("\nPreview (first 10 rows):")
print(summary.head(10).to_string(index=False))