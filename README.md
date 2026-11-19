 Automated Scoring of Children's Hand-Drawn Shapes

   Author: Yawar Sadiq  
  Institution: Federal Urdu University, Pakistan  
 Supervisor: Prof. Nam (via Syed Ali Naqi Raza, Soonchunhyang University, South Korea)  
Date: November 2026

 Overview

This project develops an automated system to predict expert scores for children's hand-drawn shapes from the BOT-2 fine motor integration task. It implements **three approaches (Machine Learning, Deep Learning, Rule-Based) and compares **seven model variants** on 8 shape categories.

## Results Summary

| Model | Mean Accuracy | Mean MAE |
|---|---|---|
| **B2 — Per-shape CNN (Best)** | **76.0%** | **0.30** |
| B-augmented — CNN + Augmentation | 75.5% | 0.30 |
| B1 — Global CNN | 71.8% | 0.38 |
| A — Random Forest (17 features) | 71.3% | 0.38 |
| A+ — Random Forest (21 features) | 71.0% | 0.38 |
| A-weighted — Random Forest + class weights | 56.9% | 0.60 |
| C — Rule-Based Scoring | 24.1% | 1.44 |

Per-shape ±1 accuracy (best model, B2): 93%–100% across all 8 shapes.

 Project Structure
ShapeScoringProject/
├── src/ # All Python scripts (23 files)
│ ├── create_metadata.py # Parse filenames → dataset_metadata.csv
│ ├── colour_background_diagnostic.py # Detect brightness confound
│ ├── inspect_shape_folders.py # Flag wrong-shape candidates (v1)
│ ├── build_clean_manifest_v3.py # Final flagging rules
│ ├── build_final_manifest.py # Apply review decisions
│ ├── generate_contact_sheets.py # Produce 21 visual inspection sheets
│ ├── prepare_review_workbook.py # Create review CSV
│ ├── split_clean_data.py # 80/20 stratified split
│ ├── extract_features_clean.py # Extract 17 base features
│ ├── add_geometry_features.py # Add 4 geometry features (→ 21 total)
│ ├── prepare_cnn_data.py # Convert to .npz for CNN
│ ├── train_approach_A.py # Random Forest (17 features)
│ ├── train_approach_A_plus.py # Random Forest (21 features)
│ ├── train_approach_A_weighted.py # Random Forest + class weights
│ ├── train_approach_B.py # CNN (global + per-shape)
│ ├── train_approach_B_augmented.py # CNN with data augmentation
│ ├── train_approach_C_rules.py # Rule-based scoring
│ └── test_one_image.py # Single-image prediction demo
├── outputs/
│ ├── tables/ # All CSVs (metrics, features, splits)
│ ├── models/ # Trained models (.joblib, .keras)
│ ├── graphs/ # 56 confusion matrices
│ └── contact_sheets/ # 21 inspection sheets
└── README.md # This file


 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
## How to Reproduce the Results

### Step 1 — Dataset Preparation

The dataset is **confidential** and not included in this repository. Place the original `Shapes/` folder in the path defined in `src/create_metadata.py` (edit the path if needed).

### Step 2 — Build metadata and inspect

python src/create_metadata.py
python src/colour_background_diagnostic.py
python src/build_clean_manifest_v3.py
python src/build_final_manifest.py

### Step 3 — Split and extract features

python src/split_clean_data.py
python src/extract_features_clean.py
python src/add_geometry_features.py
python src/prepare_cnn_data.py

### Step 4 — Train all models

python src/train_approach_A.py
python src/train_approach_A_plus.py
python src/train_approach_A_weighted.py
python src/train_approach_B.py
python src/train_approach_B_augmented.py
python src/train_approach_C_rules.py

All metrics are saved in `outputs/tables/`. All confusion matrices are saved in `outputs/graphs/`.

### Step 5 — Test on a single image

python src/test_one_image.py "path/to/image.png" "Circle"
## Methodology Summary

### Dataset

- **Original:** 4,299 images across 8 shapes
- **After inspection:** 4,286 images (13 wrong-shape excluded)
- **Split:** 3,425 train / 857 test (stratified by shape+score)

### Features (21 total)

| Group | Features |
|---|---|
| Shape (7) | circularity, solidity, extent, aspect_ratio, contour_area, perimeter, num_contours |
| Hu Moments (7) | hu_moment_1 … hu_moment_7 |
| Colour/Brightness (2) | paper_brightness, ink_darkness |
| Ink Coverage (1) | ink_ratio |
| Closure (1) | closure |
| Symmetry (1) | symmetry |
| Angles (2) | num_angles, mean_angle |

### Models

- **A (Random Forest, 17 features)** — 8 models, one per shape
- **A+ (Random Forest, 21 features)** — same, with geometry features
- **A-weighted** — sample weighting via `compute_sample_weight`
- **B1 (Global CNN)** — 1 CNN trained on all 8 shapes
- **B2 (Per-shape CNNs)** — 8 CNNs, one per shape
- **B-augmented** — B2 + random flip/rotation/zoom/brightness
- **C (Rule-based)** — rubric-derived rules (baseline)

### Evaluation

- **MAE** (mean absolute error)
- **Exact accuracy**
- **±1 accuracy**
- **Macro F1**, **Weighted F1**
## Key Findings

1. **Per-shape CNNs (B2) achieved the best results:** 76% mean accuracy, 0.30 MAE, 98% ±1 accuracy.
2. **Rule-based scoring failed** (24% mean accuracy) — confirming machine learning is necessary.
3. **Adding 4 geometry features to RF did not improve** performance.
4. **Class weighting and data augmentation did not help** at this dataset size.
5. **A brightness confound** exists in the dataset (dimmer paper correlates with lower scores) — documented and handled via explicit features.
## Limitations

- No participant-independent split (participant IDs not confirmed)
- Custom small CNN instead of ResNet/MobileNet/EfficientNet (CPU-only hardware)
- Brightness confound documented but not corrected
- Rule-based thresholds are approximations (exact rubric not provided)
- Small dataset (~3,425 training images)
## Confidentiality

The dataset is strictly confidential and has not been included in this repository. Only source code, metrics, and trained models are provided.
## Contact

- **Yawar Sadiq** — BS Computer Science, Federal Urdu University, Pakistan
- Email: yawarsadiq786@gmail.com
## License

This work was created for academic evaluation under Prof. Nam (Soonchunhyang University) and should not be redistributed without permission.


## Google Colab Notebooks (GPU Execution)

Due to hardware constraints (the local laptop has no dedicated GPU), the Deep Learning models were trained on Google Colab's NVIDIA T4 GPU. The notebooks containing the full training logs and end-to-end pipeline can be found in the root directory:

- `Colab_ResNet_Pipeline.ipynb`: Trains ResNet-50 (86.5%) and builds the End-to-End Pipeline.
- `Colab_MobileNet_DenseNet.ipynb`: Trains MobileNetV2 (98.47%) and DenseNet121 (98.36%).
- `Colab_Scoring_CNN_Retrained.ipynb`: Trains a deeper 4-block custom CNN for scoring (98.47%).