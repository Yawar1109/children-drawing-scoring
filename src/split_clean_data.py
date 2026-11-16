# ============================================================
# split_clean_data.py
# Purpose: Split the CLEAN manifest (4,286 images) into
#          train (80%) and test (20%), stratified by shape+score.
#          Uses a fixed seed for reproducibility.
# Author: Yawar Sadiq (mentored step-by-step)
# ============================================================

import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT  = r"C:\Users\dell\Documents\ShapeScoringProject"
FINAL_MANIFEST = os.path.join(PROJECT_ROOT, "outputs", "tables", "final_clean_manifest.csv")
TRAIN_CSV      = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_clean.csv")
TEST_CSV       = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_clean.csv")

# ---- load ----
print("Loading clean manifest...")
df = pd.read_csv(FINAL_MANIFEST)
print(f"Total rows: {len(df)}")

# Drop rows without a valid score or filename
df = df.dropna(subset=["score", "filename"])
df["score"] = df["score"].astype(int)

# Build a combined label for stratification
df["stratify_label"] = df["logical_shape"] + "_" + df["score"].astype(str)

# ---- check for ultra-rare classes (count < 2) ----
class_counts = df["stratify_label"].value_counts()
rare = class_counts[class_counts < 2]
if len(rare) > 0:
    print("\nUltra-rare classes (count < 2) — will be removed for stratification:")
    print(rare.to_string())
    df = df[~df["stratify_label"].isin(rare.index)]
    print(f"Remaining rows after removing rare classes: {len(df)}")
else:
    print("\nNo ultra-rare classes found — all strata are safe.")

# ---- split 80/20 with fixed seed ----
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(sss.split(df, df["stratify_label"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

# ---- report ----
print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)
print(f"Train: {len(train_df)} images")
print(f"Test : {len(test_df)} images")
print(f"\nTrain per shape:")
print(train_df["logical_shape"].value_counts().sort_index().to_string())
print(f"\nTest per shape:")
print(test_df["logical_shape"].value_counts().sort_index().to_string())

# Verify the split is balanced per shape+score
print("\nPer-stratum check (should match closely):")
train_counts = train_df["stratify_label"].value_counts().sort_index()
test_counts  = test_df["stratify_label"].value_counts().sort_index()
for label in train_counts.index[:15]:  # show first 15
    t = train_counts.get(label, 0)
    s = test_counts.get(label, 0)
    print(f"  {label:25s} train={t:4d}  test={s:4d}")

print("\n" + "=" * 60)
print("Saved:")
print(f"  Train: {TRAIN_CSV}")
print(f"  Test : {TEST_CSV}")
print("=" * 60)