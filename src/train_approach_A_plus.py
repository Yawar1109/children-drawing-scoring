# ============================================================
# train_approach_A_plus.py
# Purpose: Retrain Random Forest (per-shape) with the 21
#          extended features (17 original + closure, symmetry,
#          num_angles, mean_angle).
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix, f1_score
import joblib

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
TRAIN_FEAT = os.path.join(PROJECT_ROOT, "outputs", "tables", "train_features_plus.csv")
TEST_FEAT  = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_features_plus.csv")
METRICS_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "approach_A_plus_metrics.csv")
GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "graphs")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "models")
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "ink_ratio", "contour_area", "perimeter", "circularity",
    "solidity", "extent", "aspect_ratio",
    "hu_moment_1", "hu_moment_2", "hu_moment_3", "hu_moment_4",
    "hu_moment_5", "hu_moment_6", "hu_moment_7",
    "num_contours", "paper_brightness", "ink_darkness",
    "closure", "symmetry", "num_angles", "mean_angle",
]

SHAPES = ["Circle", "Diamond", "Overlapped Circles",
          "Overlapped Pencils", "Square", "Star", "Triangle", "Wave"]

def main():
    print("Loading extended features (21)...")
    train = pd.read_csv(TRAIN_FEAT)
    test  = pd.read_csv(TEST_FEAT)
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Features: {len(FEATURE_COLS)}")

    all_metrics = []
    for shape in SHAPES:
        print(f"\n=== {shape} ===")
        tr = train[train["logical_shape"] == shape].reset_index(drop=True)
        te = test[test["logical_shape"] == shape].reset_index(drop=True)
        if len(tr) < 5 or len(te) < 2:
            continue

        X_tr = tr[FEATURE_COLS].values
        y_tr = tr["score"].values.astype(int)
        X_te = te[FEATURE_COLS].values
        y_te = te["score"].values.astype(int)
        print(f"  Train: {len(tr)} | Test: {len(te)}")

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        y_pred_round = np.clip(np.round(y_pred).astype(int), 0, int(y_tr.max()))

        mae = mean_absolute_error(y_te, y_pred)
        acc = float((y_pred_round == y_te).mean())
        acc_pm1 = float((np.abs(y_pred_round - y_te) <= 1).mean())
        classes = sorted(set(y_te))
        f1m = f1_score(y_te, y_pred_round, labels=classes, average="macro", zero_division=0)
        f1w = f1_score(y_te, y_pred_round, labels=classes, average="weighted", zero_division=0)

        print(f"  MAE:            {mae:.4f}")
        print(f"  Accuracy:       {acc:.4f}")
        print(f"  Accuracy (+/-1):{acc_pm1:.4f}")
        print(f"  F1 macro:       {f1m:.4f}")
        print(f"  F1 weighted:    {f1w:.4f}")

        # Save model
        joblib.dump(rf, os.path.join(MODELS_DIR, f"rf_plus_{shape.replace(' ', '_')}.joblib"))

        # Confusion matrix
        labels = list(range(0, int(max(y_tr.max(), y_te.max())) + 1))
        cm = confusion_matrix(y_te, y_pred_round, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Purples")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"A+ — {shape}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(GRAPHS_DIR, f"confusion_Aplus_{shape.replace(' ', '_')}.png"), dpi=100)
        plt.close(fig)

        all_metrics.append({
            "shape": shape, "n_train": len(tr), "n_test": len(te),
            "mae": round(mae, 4), "accuracy": round(acc, 4),
            "accuracy_pm1": round(acc_pm1, 4),
            "f1_macro": round(f1m, 4), "f1_weighted": round(f1w, 4),
        })

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print("\n" + "=" * 60)
    print("APPROACH A+ (21 features) — FINAL METRICS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {METRICS_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()