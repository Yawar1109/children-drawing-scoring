# ============================================================
# prepare_review_workbook.py
# Purpose: Create a review CSV with one row per suspect.
# ============================================================

import os
import pandas as pd

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
SUSPECTS_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "suspected_images_v3_PRIVATE.csv")
REVIEW_CSV   = os.path.join(PROJECT_ROOT, "outputs", "tables", "review_decisions.csv")

df = pd.read_csv(SUSPECTS_CSV)
# Add empty 'decision' column
df["decision"] = ""
# Keep only relevant columns
out = df[["suspect_id", "filename", "logical_shape", "decision"]]
out.to_csv(REVIEW_CSV, index=False)

print(f"Created: {REVIEW_CSV}")
print(f"Total rows: {len(out)}")
print("\nFirst 5 rows:")
print(out.head().to_string(index=False))