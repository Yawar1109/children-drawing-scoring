# ============================================================
# build_final_manifest.py
# Purpose: Build final clean manifest from expert review.
# ============================================================

import os
import pandas as pd

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
SUSPECTS_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "suspected_images_v3_PRIVATE.csv")
MANIFEST_V3  = os.path.join(PROJECT_ROOT, "outputs", "tables", "clean_manifest_v3.csv")
FINAL_CSV    = os.path.join(PROJECT_ROOT, "outputs", "tables", "final_clean_manifest.csv")
REVIEW_LOG   = os.path.join(PROJECT_ROOT, "outputs", "tables", "review_log_PRIVATE.csv")

# ------------------------------------------------------------
# The expert review (based on visual inspection of all 21 sheets)
# ------------------------------------------------------------
EXCLUDE_IDS = {
    "SUS_0031",  # Diamond folder → star
    "SUS_0052",  # Overlapped Circles → scribble
    "SUS_0074",  # Overlapped Circles → single wave
    "SUS_0081",  # Overlapped Circles → single wave
    "SUS_0164",  # Star → triangle
    "SUS_0177",  # Star → arrow/triangle
    "SUS_0185",  # Star → partial shape
    "SUS_0214",  # Star → scribble
    "SUS_0240",  # Wave → triangle
    "SUS_0242",  # Wave → triangle
    "SUS_0259",  # Wave → triangle
    "SUS_0334",  # Wave → triangle
    "SUS_0336",  # Wave → triangle
}

UNCERTAIN_IDS = {
    "SUS_0085",  # Overlapped Circles → one elongated shape
    "SUS_0217",  # Star → two overlapping shapes
    "SUS_0171",  # Star → two overlapping shapes
}

if __name__ == "__main__":
    print("Loading data...")
    suspects = pd.read_csv(SUSPECTS_CSV)
    manifest = pd.read_csv(MANIFEST_V3)

    # 1. Build decision map
    decisions = {}
    for _, row in suspects.iterrows():
        sid = row["suspect_id"]
        if sid in EXCLUDE_IDS:
            decisions[sid] = "EXCLUDE"
        elif sid in UNCERTAIN_IDS:
            decisions[sid] = "UNCERTAIN"
        else:
            decisions[sid] = "KEEP"

    # 2. Attach decision to suspects
    suspects["decision"] = suspects["suspect_id"].map(decisions)

    # 3. Save review log (PRIVATE — stays on your computer)
    suspects[["suspect_id", "filename", "logical_shape", "decision"]].to_csv(
        REVIEW_LOG, index=False)
    print(f"Saved review log (PRIVATE): {REVIEW_LOG}")

    # 4. Build set of filenames to exclude
    excluded_files = set(suspects[suspects["decision"] == "EXCLUDE"]["filename"])

    # 5. Final manifest = all rows in v3 EXCEPT excluded
    final = manifest[~manifest["filename"].isin(excluded_files)].copy()

    # 6. Add a column marking uncertain (for transparency in report)
    uncertain_files = set(suspects[suspects["decision"] == "UNCERTAIN"]["filename"])
    final["is_uncertain"] = final["filename"].isin(uncertain_files).astype(int)

    # 7. Save
    final.to_csv(FINAL_CSV, index=False)

    # 8. Report
    print("\n" + "=" * 60)
    print("REVIEW SUMMARY")
    print("=" * 60)
    print(f"Total suspects reviewed:  {len(suspects)}")
    print(f"  KEEP:                    {(suspects['decision'] == 'KEEP').sum()}")
    print(f"  EXCLUDE:                 {(suspects['decision'] == 'EXCLUDE').sum()}")
    print(f"  UNCERTAIN:               {(suspects['decision'] == 'UNCERTAIN').sum()}")
    print(f"\nTotal images in v3 manifest: {len(manifest)}")
    print(f"Total images in FINAL:       {len(final)}")
    print(f"  ({len(excluded_files)} excluded)")
    print(f"\nSaved FINAL manifest: {FINAL_CSV}")
    print("=" * 60)