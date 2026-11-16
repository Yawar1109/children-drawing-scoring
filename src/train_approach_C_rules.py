# ============================================================
# train_approach_C_rules.py
# Purpose: Rule-based scoring (Approach C).
#          Each rubric item is a fixed rule.
#          Sum the rubric items → predicted score.
#          No training — pure rules applied to test set.
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
TEST_FEAT = os.path.join(PROJECT_ROOT, "outputs", "tables", "test_features_plus.csv")
METRICS_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "approach_C_metrics.csv")
GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "graphs")

SHAPES = ["Circle", "Diamond", "Overlapped Circles",
          "Overlapped Pencils", "Square", "Star", "Triangle", "Wave"]

# ---- Maximum score per shape (from rubric) ----
MAX_SCORE = {
    "Circle": 4, "Square": 5, "Triangle": 5, "Diamond": 5,
    "Wave": 4, "Star": 5, "Overlapped Circles": 6, "Overlapped Pencils": 6,
}

def rule_score_circle(row):
    score = 0
    if row["circularity"] > 0.4: score += 1            # basic shape
    if row["closure"] == 1: score += 1                 # closure
    if 0.5 <= row["aspect_ratio"] <= 2.0: score += 1   # edges similar
    if row["contour_area"] > 200: score += 1           # overall size
    return min(score, MAX_SCORE["Circle"])

def rule_score_square(row):
    score = 0
    if row["num_angles"] >= 3 and row["num_angles"] <= 5: score += 1  # 4 sides
    if row["closure"] == 1: score += 1
    if row["extent"] > 0.4: score += 1                 # similar edges (approx)
    if 0.7 <= row["aspect_ratio"] <= 1.4: score += 1   # orientation
    if row["contour_area"] > 200: score += 1
    return min(score, MAX_SCORE["Square"])

def rule_score_triangle(row):
    score = 0
    if 2 <= row["num_angles"] <= 4: score += 1
    if row["closure"] == 1: score += 1
    if row["extent"] > 0.4: score += 1
    if 0.5 <= row["aspect_ratio"] <= 2.0: score += 1
    return min(score, MAX_SCORE["Triangle"])

def rule_score_diamond(row):
    score = 0
    if 3 <= row["num_angles"] <= 5: score += 1
    if row["closure"] == 1: score += 1
    if row["extent"] > 0.4: score += 1
    if 0.7 <= row["aspect_ratio"] <= 1.4: score += 1
    if row["contour_area"] > 200: score += 1
    return min(score, MAX_SCORE["Diamond"])

def rule_score_wave(row):
    score = 0
    if row["circularity"] < 0.3 and row["num_angles"] < 6: score += 1
    if row["extent"] > 0.3: score += 1
    if 0.5 <= row["aspect_ratio"] <= 2.0: score += 1
    if row["contour_area"] > 200: score += 1
    return min(score, MAX_SCORE["Wave"])

def rule_score_star(row):
    score = 0
    if 8 <= row["num_angles"] <= 14: score += 1
    if row["closure"] == 1: score += 1
    if row["extent"] > 0.3: score += 1
    if 0.5 <= row["aspect_ratio"] <= 2.0: score += 1
    if row["contour_area"] > 200: score += 1
    return min(score, MAX_SCORE["Star"])

def rule_score_overlapped(row, max_sc):
    score = 0
    if row["num_contours"] >= 2: score += 1         # basic shape (two objects)
    if row["closure"] == 1: score += 1
    if row["extent"] > 0.3: score += 1              # edges similar
    if row["contour_area"] > 200: score += 1
    return min(score, max_sc)

def apply_rules(row):
    s = row["logical_shape"]
    if s == "Circle":              return rule_score_circle(row)
    if s == "Square":              return rule_score_square(row)
    if s == "Triangle":            return rule_score_triangle(row)
    if s == "Diamond":             return rule_score_diamond(row)
    if s == "Wave":                return rule_score_wave(row)
    if s == "Star":                return rule_score_star(row)
    if s == "Overlapped Circles":  return rule_score_overlapped(row, MAX_SCORE["Overlapped Circles"])
    if s == "Overlapped Pencils":  return rule_score_overlapped(row, MAX_SCORE["Overlapped Pencils"])
    return 0

if __name__ == "__main__":
    print("Loading test features (21)...")
    test = pd.read_csv(TEST_FEAT)
    print(f"Test rows: {len(test)}")

    test["predicted"] = test.apply(apply_rules, axis=1)

    metrics = []
    for shape in SHAPES:
        sub = test[test["logical_shape"] == shape]
        if len(sub) < 2:
            continue
        y_te = sub["score"].values.astype(int)
        y_pr = sub["predicted"].values.astype(int)

        mae = mean_absolute_error(y_te, y_pr)
        acc = float((y_pr == y_te).mean())
        acc_pm1 = float((np.abs(y_pr - y_te) <= 1).mean())
        classes = sorted(set(y_te))
        f1m = f1_score(y_te, y_pr, labels=classes, average="macro", zero_division=0)
        f1w = f1_score(y_te, y_pr, labels=classes, average="weighted", zero_division=0)

        print(f"\n=== {shape} ===")
        print(f"  n_test={len(sub)}  MAE={mae:.4f}  acc={acc:.4f}  acc±1={acc_pm1:.4f}  f1w={f1w:.4f}")

        labels = list(range(0, MAX_SCORE[shape] + 1))
        cm = confusion_matrix(y_te, y_pr, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Oranges")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"C-rules — {shape}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(GRAPHS_DIR, f"confusion_C_{shape.replace(' ', '_')}.png"), dpi=100)
        plt.close(fig)

        metrics.append({
            "shape": shape, "n_test": int(len(sub)),
            "mae": round(mae, 4), "accuracy": round(acc, 4),
            "accuracy_pm1": round(acc_pm1, 4),
            "f1_macro": round(f1m, 4), "f1_weighted": round(f1w, 4),
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print("\n" + "=" * 60)
    print("APPROACH C (rule-based) — FINAL METRICS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {METRICS_CSV}")
    print("=" * 60)