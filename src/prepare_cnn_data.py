# ============================================================
# prepare_cnn_data.py
# Purpose: Convert the clean train/test splits into NumPy
#          arrays (.npz) for fast CNN training.
# ============================================================

import os
import numpy as np
import cv2
import pandas as pd

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
TRAIN_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_clean.csv")
TEST_CSV  = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_clean.csv")
METADATA_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")
TRAIN_NPZ = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_images.npz")
TEST_NPZ  = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_images.npz")

SHAPES = ["Circle", "Diamond", "Overlapped Circles",
          "Overlapped Pencils", "Square", "Star", "Triangle", "Wave"]

IMG_SIZE = 128

def build_arrays(meta_csv, split_name):
    print(f"\n=== Building {split_name} arrays ===")
    df = pd.read_csv(meta_csv)
    meta = pd.read_csv(METADATA_CSV)
    path_lookup = dict(zip(meta["filename"], meta["image_path"]))
    df["image_path"] = df["filename"].map(path_lookup)
    df = df.dropna(subset=["image_path"])
    print(f"Rows with valid image_path: {len(df)}")

    imgs = []
    scores = []
    shape_idx = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 500 == 0 or i == 1:
            print(f"  [{split_name}] {i}/{len(df)}...")
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Add channel dimension: (H, W) -> (H, W, 1)
        img = np.expand_dims(img, axis=-1)
        imgs.append(img)
        scores.append(int(row["score"]))
        shape_idx.append(SHAPES.index(row["logical_shape"]))

    X = np.stack(imgs)
    y_score = np.array(scores, dtype=np.int32)
    y_shape = np.array(shape_idx, dtype=np.int32)

    print(f"X shape: {X.shape} | y_score: {y_score.shape} | y_shape: {y_shape.shape}")
    return X, y_score, y_shape

if __name__ == "__main__":
    print("Building CNN data arrays...")
    X_tr, y_tr_score, y_tr_shape = build_arrays(TRAIN_CSV, "train")
    X_te, y_te_score, y_te_shape = build_arrays(TEST_CSV, "test")

    np.savez_compressed(TRAIN_NPZ, X=X_tr, y_score=y_tr_score, y_shape=y_tr_shape)
    np.savez_compressed(TEST_NPZ,  X=X_te, y_score=y_te_score, y_shape=y_te_shape)

    print("\n" + "=" * 60)
    print("DONE — CNN data ready")
    print(f"Train: {TRAIN_NPZ}")
    print(f"Test : {TEST_NPZ}")
    print(f"Train shape: {X_tr.shape}")
    print(f"Test shape : {X_te.shape}")
    print("=" * 60)