# ============================================================
# build_clean_manifest.py
# Purpose: Re-compute clean 4-feature values for all 4,299 images,
#          apply data-driven outlier rules per shape, and save
#          three CSVs: clean_manifest, inspection_summary, and
#          a private suspects list.
# Author: Yawar Sadiq (mentored step-by-step)
# ============================================================

import os
import math
import numpy as np
import cv2
import pandas as pd

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
METADATA_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "clean_manifest.csv")
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "outputs", "tables", "inspection_summary.csv")
SUSPECTS_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "suspected_images_PRIVATE.csv")

# ------------------------------------------------------------
# 2. Feature computation (same as check_distribution.py)
# ------------------------------------------------------------
def measure_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize and crop border
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img[12:-12, 12:-12]

    # Noise reduction and threshold
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img_blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to connect broken strokes
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {"circularity": 0.0, "solidity": 0.0, "extent": 0.0,
                "num_contours": 0}

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    if perimeter == 0 or area == 0:
        return {"circularity": 0.0, "solidity": 0.0, "extent": 0.0,
                "num_contours": len(contours)}

    circularity = (4 * math.pi * area) / (perimeter ** 2)

    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    x, y, w, h = cv2.boundingRect(largest)
    bbox_area = w * h
    extent = area / bbox_area if bbox_area > 0 else 0.0

    return {"circularity": circularity, "solidity": solidity,
            "extent": extent, "num_contours": len(contours)}

# ------------------------------------------------------------
# 3. Shape-specific signature rules
# ------------------------------------------------------------
def check_signature_rules(shape, circ, solidity, num_contours):
    reasons = []
    if num_contours == 0:
        reasons.append("empty_drawing")
    if shape == "Wave" and circ > 0.40:
        reasons.append("wave_signature_violation")
    if shape == "Star" and solidity > 0.75:
        reasons.append("star_signature_violation")
    if shape == "Circle" and solidity > 0.98 and circ < 0.15:
        reasons.append("circle_signature_violation")
    return reasons

# ------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Loading metadata...")
    df = pd.read_csv(METADATA_CSV).dropna(subset=["score"])
    df["score"] = df["score"].astype(int)
    print(f"Total images: {len(df)}")

    # --- 4.1 Compute features for all images ---
    print("Computing clean features for all images (~1 minute)...")
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i % 1000 == 0:
            print(f"  Processed {i}...")
        feats = measure_image(row["image_path"])
        if feats is None:
            continue
        feats["filename"] = row["filename"]
        feats["logical_shape"] = row["logical_shape"]
        feats["score"] = int(row["score"])
        rows.append(feats)

    per_image = pd.DataFrame(rows)
    print(f"Feature computation done: {len(per_image)} images")

    # --- 4.2 Compute per-shape percentiles ---
    print("\nComputing per-shape percentiles...")
    pct_rows = []
    for shape, group in per_image.groupby("logical_shape"):
        circ = group["circularity"]
        sol  = group["solidity"]
        ext  = group["extent"]
        pct_rows.append({
            "shape":       shape,
            "circ_p05":    circ.quantile(0.05),
            "circ_p95":    circ.quantile(0.95),
            "sol_p05":     sol.quantile(0.05),
            "sol_p95":     sol.quantile(0.95),
            "ext_p05":     ext.quantile(0.05),
            "ext_p95":     ext.quantile(0.95),
        })
    pct_df = pd.DataFrame(pct_rows).set_index("shape")
    print(pct_df.round(3).to_string())

    # --- 4.3 Apply rules ---
    print("\nApplying flagging rules...")
    def flags_for_row(row):
        shape = row["logical_shape"]
        circ  = row["circularity"]
        sol   = row["solidity"]
        ext   = row["extent"]
        nc    = row["num_contours"]
        p     = pct_df.loc[shape]

        outlier_features = 0
        if circ < p["circ_p05"] or circ > p["circ_p95"]:
            outlier_features += 1
        if sol < p["sol_p05"] or sol > p["sol_p95"]:
            outlier_features += 1
        if ext < p["ext_p05"] or ext > p["ext_p95"]:
            outlier_features += 1

        reasons = []
        if outlier_features >= 2:
            reasons.append(f"multi_feature_outlier_{outlier_features}")
        reasons.extend(check_signature_rules(shape, circ, sol, nc))

        return reasons

    per_image["flag_reasons_list"] = per_image.apply(flags_for_row, axis=1)
    per_image["flag_reasons"] = per_image["flag_reasons_list"].apply(
        lambda lst: ";".join(lst))
    per_image["suggested_review"] = per_image["flag_reasons_list"].apply(
        lambda lst: "yes" if len(lst) > 0 else "no")

    # --- 4.4 Build the clean manifest ---
    manifest_cols = ["filename", "logical_shape", "score",
                     "circularity", "solidity", "extent", "num_contours",
                     "suggested_review", "flag_reasons"]
    manifest = per_image[manifest_cols].round(4)
    manifest.to_csv(MANIFEST_CSV, index=False)
    print(f"\nSaved manifest:   {MANIFEST_CSV}")

    # --- 4.5 Summary per shape ---
    summary_rows = []
    for shape, group in manifest.groupby("logical_shape"):
        n_total = len(group)
        n_flag = (group["suggested_review"] == "yes").sum()
        pct = round(100.0 * n_flag / n_total, 1) if n_total > 0 else 0.0
        summary_rows.append({
            "logical_shape": shape,
            "total_images": n_total,
            "flagged_for_review": int(n_flag),
            "flagged_percent": pct,
        })
    summary = pd.DataFrame(summary_rows).sort_values("logical_shape")
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary:    {SUMMARY_CSV}")
    print("\nSummary per shape:")
    print(summary.to_string(index=False))

    # --- 4.6 Private suspects list (real filenames) ---
    suspects = manifest[manifest["suggested_review"] == "yes"].copy()
    suspects[["filename", "logical_shape", "score",
              "flag_reasons"]].to_csv(SUSPECTS_CSV, index=False)
    print(f"\nSaved suspects list (PRIVATE): {SUSPECTS_CSV}")
    print(f"Total suspects flagged: {len(suspects)}")

    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)