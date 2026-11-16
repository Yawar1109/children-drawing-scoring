# ============================================================
# train_approach_B_augmented.py
# Purpose: CNN with data augmentation (per-shape models).
#          Random flip, rotation, zoom, brightness.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, confusion_matrix, f1_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
TRAIN_NPZ = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_images.npz")
TEST_NPZ  = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_images.npz")
METRICS_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "approach_B_augmented_metrics.csv")
GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "graphs")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "models")
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SHAPES = ["Circle", "Diamond", "Overlapped Circles",
          "Overlapped Pencils", "Square", "Star", "Triangle", "Wave"]

print("Loading CNN data...")
tr = np.load(TRAIN_NPZ); te = np.load(TEST_NPZ)
X_tr = tr["X"]; y_tr = tr["y_score"]; s_tr = tr["y_shape"]
X_te = te["X"]; y_te = te["y_score"]; s_te = te["y_shape"]
print(f"Train: {X_tr.shape}  Test: {X_te.shape}")

# ------------------------------------------------------------
# Augmentation pipeline (Keras preprocessing layers)
# ------------------------------------------------------------
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),          # ~±18 degrees
    layers.RandomZoom(0.10),
    layers.RandomBrightness(0.10),
], name="augmentation")

def build_cnn(n_classes):
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 1)),
        data_aug,
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------------------------------------
# Train per-shape with augmentation
# ------------------------------------------------------------
all_metrics = []
for idx, shape in enumerate(SHAPES):
    print(f"\n=== {shape} ===")
    tr_mask = (s_tr == idx); te_mask = (s_te == idx)
    if tr_mask.sum() < 10 or te_mask.sum() < 2:
        print(f"  Skipping (train={tr_mask.sum()}, test={te_mask.sum()})")
        continue

    Xtr_s = X_tr[tr_mask]; ytr_s = y_tr[tr_mask]
    Xte_s = X_te[te_mask]; yte_s = y_te[te_mask]

    unique_scores = sorted(set(ytr_s) | set(yte_s))
    score_to_local = {s: i for i, s in enumerate(unique_scores)}
    local_to_score = {i: s for i, s in enumerate(unique_scores)}
    n_local = len(unique_scores)
    ytr_local = np.array([score_to_local[s] for s in ytr_s])
    yte_local = np.array([score_to_local[s] for s in yte_s])

    print(f"  n_train={len(Xtr_s)}  n_test={len(Xte_s)}  classes={n_local}")

    m = build_cnn(n_local)
    m.fit(Xtr_s, ytr_local, validation_split=0.15,
          epochs=15, batch_size=16, verbose=0)
    m.save(os.path.join(MODELS_DIR, f"cnn_aug_{shape.replace(' ', '_')}.keras"))

    pred_local = m.predict(Xte_s, verbose=0).argmax(axis=1)
    pred_real = np.array([local_to_score[p] for p in pred_local])

    mae = mean_absolute_error(yte_s, pred_real)
    acc = float((pred_real == yte_s).mean())
    acc_pm1 = float((np.abs(pred_real - yte_s) <= 1).mean())
    classes = sorted(set(yte_s))
    f1m = f1_score(yte_s, pred_real, labels=classes, average="macro", zero_division=0)
    f1w = f1_score(yte_s, pred_real, labels=classes, average="weighted", zero_division=0)

    print(f"  MAE={mae:.4f}  acc={acc:.4f}  acc±1={acc_pm1:.4f}  f1w={f1w:.4f}")

    labels = list(range(min(unique_scores), max(unique_scores) + 1))
    cm = confusion_matrix(yte_s, pred_real, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"B-aug — {shape}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, f"confusion_Baug_{shape.replace(' ', '_')}.png"), dpi=100)
    plt.close(fig)

    all_metrics.append({
        "shape": shape, "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
        "mae": round(mae, 4), "accuracy": round(acc, 4),
        "accuracy_pm1": round(acc_pm1, 4),
        "f1_macro": round(f1m, 4), "f1_weighted": round(f1w, 4),
    })

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(METRICS_CSV, index=False)
print("\n" + "=" * 60)
print("APPROACH B-AUGMENTED — FINAL METRICS")
print("=" * 60)
print(metrics_df.to_string(index=False))
print(f"\nSaved: {METRICS_CSV}")
print("=" * 60)