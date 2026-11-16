# ============================================================
# train_approach_B.py  (Option 3 — both variants)
# Purpose: Train TWO CNN variants on the same clean data:
#          (B1) 1 global CNN trained on all 8 shapes together
#          (B2) 8 per-shape CNNs (matches Approach A fairly)
#          Evaluate both per-shape, save all metrics.
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF INFO logs

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
TRAIN_NPZ    = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_images.npz")
TEST_NPZ     = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_images.npz")
METRICS_GLOBAL = os.path.join(PROJECT_ROOT, "outputs", "tables", "approach_B_global_metrics.csv")
METRICS_PSHAPE = os.path.join(PROJECT_ROOT, "outputs", "tables", "approach_B_pershape_metrics.csv")
GRAPHS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "graphs")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "models")
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SHAPES = ["Circle", "Diamond", "Overlapped Circles",
          "Overlapped Pencils", "Square", "Star", "Triangle", "Wave"]

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
print("Loading CNN data...")
tr = np.load(TRAIN_NPZ); te = np.load(TEST_NPZ)
X_tr = tr["X"]; y_tr = tr["y_score"]; s_tr = tr["y_shape"]
X_te = te["X"]; y_te = te["y_score"]; s_te = te["y_shape"]
print(f"Train: {X_tr.shape}  Test: {X_te.shape}")

MAX_SCORE = int(max(y_tr.max(), y_te.max()))
N_CLASSES = MAX_SCORE + 1
print(f"Score classes: {N_CLASSES} (0 to {MAX_SCORE})")

# ------------------------------------------------------------
# Build CNN
# ------------------------------------------------------------
def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ------------------------------------------------------------
# Evaluate per shape
# ------------------------------------------------------------
def evaluate_per_shape(y_true, y_pred_class, shape_true, prefix, csv_out):
    metrics = []
    for idx, shape in enumerate(SHAPES):
        mask = (shape_true == idx)
        if mask.sum() < 2:
            continue
        yt = y_true[mask]; yp = y_pred_class[mask]
        mae = mean_absolute_error(yt, yp)
        acc = float((yp == yt).mean())
        acc_pm1 = float((np.abs(yp - yt) <= 1).mean())
        classes = sorted(set(yt))
        f1m = f1_score(yt, yp, labels=classes, average="macro", zero_division=0)
        f1w = f1_score(yt, yp, labels=classes, average="weighted", zero_division=0)
        print(f"  {shape:22s} n={mask.sum():4d}  MAE={mae:.3f}  acc={acc:.3f}  acc±1={acc_pm1:.3f}  f1w={f1w:.3f}")
        # Confusion matrix
        labels = list(range(0, MAX_SCORE + 1))
        cm = confusion_matrix(yt, yp, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{prefix} — {shape}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(GRAPHS_DIR,
                    f"confusion_{prefix}_{shape.replace(' ', '_')}.png"), dpi=100)
        plt.close(fig)
        metrics.append({"shape": shape, "n_test": int(mask.sum()),
                        "mae": round(mae, 4), "accuracy": round(acc, 4),
                        "accuracy_pm1": round(acc_pm1, 4),
                        "f1_macro": round(f1m, 4), "f1_weighted": round(f1w, 4)})
    pd.DataFrame(metrics).to_csv(csv_out, index=False)
    return metrics

# ------------------------------------------------------------
# (B1) Global CNN
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("APPROACH B1 — GLOBAL CNN (all shapes together)")
print("=" * 60)
model_global = build_cnn()
model_global.summary()
history = model_global.fit(
    X_tr, y_tr,
    validation_split=0.15,
    epochs=15,
    batch_size=32,
    verbose=2,
)
model_global.save(os.path.join(MODELS_DIR, "cnn_global.keras"))
probs = model_global.predict(X_te, verbose=0)
pred_global = probs.argmax(axis=1)
print("\nPer-shape results (global CNN):")
evaluate_per_shape(y_te, pred_global, s_te, "B1global", METRICS_GLOBAL)

# ------------------------------------------------------------
# (B2) Per-shape CNNs
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("APPROACH B2 — PER-SHAPE CNNs")
print("=" * 60)
pershape_metrics = []
for idx, shape in enumerate(SHAPES):
    print(f"\n--- {shape} ---")
    tr_mask = (s_tr == idx); te_mask = (s_te == idx)
    if tr_mask.sum() < 10 or te_mask.sum() < 2:
        print(f"  Skipping (train={tr_mask.sum()}, test={te_mask.sum()})")
        continue

    Xtr_s = X_tr[tr_mask]; ytr_s = y_tr[tr_mask]
    Xte_s = X_te[te_mask]; yte_s = y_te[te_mask]

    # For per-shape, remap scores to 0..K-1 to shrink the output layer
    unique_scores = sorted(set(ytr_s) | set(yte_s))
    score_to_local = {s: i for i, s in enumerate(unique_scores)}
    local_to_score = {i: s for i, s in enumerate(unique_scores)}
    n_local = len(unique_scores)
    ytr_local = np.array([score_to_local[s] for s in ytr_s])
    yte_local = np.array([score_to_local[s] for s in yte_s])

    # Build per-shape CNN (smaller output layer)
    m = keras.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_local, activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(Xtr_s, ytr_local, validation_split=0.15, epochs=15,
          batch_size=16, verbose=0)
    m.save(os.path.join(MODELS_DIR, f"cnn_{shape.replace(' ', '_')}.keras"))

    pred_local = m.predict(Xte_s, verbose=0).argmax(axis=1)
    pred_real = np.array([local_to_score[p] for p in pred_local])

    mae = mean_absolute_error(yte_s, pred_real)
    acc = float((pred_real == yte_s).mean())
    acc_pm1 = float((np.abs(pred_real - yte_s) <= 1).mean())
    classes = sorted(set(yte_s))
    f1m = f1_score(yte_s, pred_real, labels=classes, average="macro", zero_division=0)
    f1w = f1_score(yte_s, pred_real, labels=classes, average="weighted", zero_division=0)
    print(f"  n_test={te_mask.sum()}  MAE={mae:.3f}  acc={acc:.3f}  acc±1={acc_pm1:.3f}  f1w={f1w:.3f}")
    pershape_metrics.append({"shape": shape, "n_test": int(te_mask.sum()),
                             "mae": round(mae, 4), "accuracy": round(acc, 4),
                             "accuracy_pm1": round(acc_pm1, 4),
                             "f1_macro": round(f1m, 4), "f1_weighted": round(f1w, 4)})

pd.DataFrame(pershape_metrics).to_csv(METRICS_PSHAPE, index=False)

print("\n" + "=" * 60)
print("APPROACH B2 — PER-SHAPE CNN METRICS")
print("=" * 60)
print(pd.DataFrame(pershape_metrics).to_string(index=False))
print(f"\nSaved:\n  {METRICS_GLOBAL}\n  {METRICS_PSHAPE}")
print("=" * 60)