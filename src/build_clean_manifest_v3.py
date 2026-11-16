# ============================================================
# build_clean_manifest_v3.py
# Purpose: Same as v2, but Rule 4 is TIGHTENED:
#   highscore_but_invisible now requires
#   circularity < 0.02 AND solidity < 0.05
# (was: < 0.05 and < 0.10)
# ============================================================

import os
import math
import numpy as np
import cv2
import pandas as pd

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
METADATA_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "clean_manifest_v3.csv")
SUMMARY_CSV  = os.path.join(PROJECT_ROOT, "outputs", "tables", "inspection_summary_v3.csv")
SUSPECTS_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "suspected_images_v3_PRIVATE.csv")

def measure_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img[12:-12, 12:-12]
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img_blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return {"circularity": 0.0, "solidity": 0.0, "extent": 0.0, "num_contours": 0}
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    if perimeter == 0 or area == 0:
        return {"circularity": 0.0, "solidity": 0.0, "extent": 0.0,
                "num_contours": len(contours)}
    circ = (4 * math.pi * area) / (perimeter ** 2)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    sol = area / hull_area if hull_area > 0 else 0.0
    x, y, w, h = cv2.boundingRect(largest)
    bbox_area = w * h
    ext = area / bbox_area if bbox_area > 0 else 0.0
    return {"circularity": circ, "solidity": sol, "extent": ext,
            "num_contours": len(contours)}

def check_rules(row, p10):
    reasons = []
    shape = row["logical_shape"]; circ = row["circularity"]
    sol = row["solidity"]; ext = row["extent"]
    nc = row["num_contours"]; score = row["score"]

    # Rule 1
    if circ < p10["circ_p10"] and sol < p10["sol_p10"] and ext < p10["ext_p10"]:
        reasons.append("very_poor_drawing")
    # Rule 2
    if shape == "Wave" and circ > 0.40:
        reasons.append("wave_signature_violation")
    if shape == "Star" and sol > 0.75:
        reasons.append("star_signature_violation")
    if shape == "Circle" and sol > 0.98 and circ < 0.15:
        reasons.append("circle_signature_violation")
    # Rule 3
    if nc == 0:
        reasons.append("empty_drawing")
    # Rule 4 — TIGHTENED
    if score >= 4 and nc == 1:
        if circ < 0.02 and sol < 0.05:
            reasons.append("highscore_but_invisible")
    return reasons

if __name__ == "__main__":
    print("Loading metadata...")
    df = pd.read_csv(METADATA_CSV).dropna(subset=["score"])
    df["score"] = df["score"].astype(int)
    print(f"Total images: {len(df)}")

    print("Computing clean features (~1 min)...")
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i % 1000 == 0: print(f"  {i}...")
        f = measure_image(row["image_path"])
        if f is None: continue
        f["filename"] = row["filename"]
        f["logical_shape"] = row["logical_shape"]
        f["score"] = int(row["score"])
        rows.append(f)
    per_image = pd.DataFrame(rows)

    print("\nComputing 10th percentiles...")
    p10_rows = []
    for shape, g in per_image.groupby("logical_shape"):
        p10_rows.append({
            "shape": shape,
            "circ_p10": g["circularity"].quantile(0.10),
            "sol_p10":  g["solidity"].quantile(0.10),
            "ext_p10":  g["extent"].quantile(0.10),
        })
    p10_df = pd.DataFrame(p10_rows).set_index("shape")
    print(p10_df.round(4).to_string())

    print("\nApplying v3 rules...")
    reasons_list = []
    for _, row in per_image.iterrows():
        p = p10_df.loc[row["logical_shape"]].to_dict()
        reasons_list.append(check_rules(row.to_dict(), p))

    per_image["flag_reasons_list"] = reasons_list
    per_image["flag_reasons"] = per_image["flag_reasons_list"].apply(lambda l: ";".join(l))
    per_image["suggested_review"] = per_image["flag_reasons_list"].apply(
        lambda l: "yes" if l else "no")

    cols = ["filename", "logical_shape", "score", "circularity", "solidity",
            "extent", "num_contours", "suggested_review", "flag_reasons"]
    per_image[cols].round(4).to_csv(MANIFEST_CSV, index=False)
    print(f"\nSaved manifest: {MANIFEST_CSV}")

    summary_rows = []
    for shape, g in per_image.groupby("logical_shape"):
        n_total = len(g); n_flag = (g["suggested_review"] == "yes").sum()
        summary_rows.append({
            "logical_shape": shape,
            "total_images": n_total,
            "flagged_for_review": int(n_flag),
            "flagged_percent": round(100.0 * n_flag / n_total, 1) if n_total else 0.0,
        })
    summary = pd.DataFrame(summary_rows).sort_values("logical_shape")
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))

    suspects = per_image[per_image["suggested_review"] == "yes"].copy().reset_index(drop=True)
    suspects["suspect_id"] = ["SUS_%04d" % (i + 1) for i in range(len(suspects))]

    # PRIVATE — real filenames
    suspects[["suspect_id", "filename", "logical_shape", "score", "flag_reasons"]].to_csv(
        SUSPECTS_CSV, index=False)
    print(f"\nSaved suspects (PRIVATE): {SUSPECTS_CSV}")
    print(f"Total suspects: {len(suspects)}")

    from collections import Counter
    counter = Counter()
    for lst in per_image["flag_reasons_list"]:
        for r in lst: counter[r] += 1
    print("\nReason breakdown:")
    for r, c in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")
    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)