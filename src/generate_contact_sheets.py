# ============================================================
# generate_contact_sheets.py
# Purpose: For each shape, split suspects into batches of 20,
#          and draw a contact sheet PNG (5 cols x 4 rows) with
#          300x300 thumbnails. Save to outputs/contact_sheets/.
# Author: Yawar Sadiq (mentored step-by-step)
# ============================================================

import os
import math
import numpy as np
import cv2
import pandas as pd

PROJECT_ROOT = r"C:\Users\dell\Documents\ShapeScoringProject"
SUSPECTS_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "suspected_images_v3_PRIVATE.csv")
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "clean_manifest_v3.csv")
METADATA_CSV = os.path.join(PROJECT_ROOT, "outputs", "tables", "dataset_metadata.csv")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "contact_sheets")

# Layout constants
THUMB_SIZE   = 300          # each thumbnail 300x300 pixels
CELL_PADDING = 8            # padding around each thumbnail
LABEL_HEIGHT = 26           # space for the SUS_XXXX label below each thumbnail
COLS         = 5            # 5 columns
ROWS         = 4            # 4 rows
BATCH_SIZE   = COLS * ROWS  # 20 per sheet
HEADER_HEIGHT = 40          # top title
FOOTER_HEIGHT = 40          # bottom footer with IDs
SHEET_WIDTH  = COLS * (THUMB_SIZE + 2 * CELL_PADDING)
CELL_HEIGHT  = THUMB_SIZE + LABEL_HEIGHT + 2 * CELL_PADDING
SHEET_HEIGHT = HEADER_HEIGHT + ROWS * CELL_HEIGHT + FOOTER_HEIGHT

# ------------------------------------------------------------
# 1. Build filename -> image_path lookup
# ------------------------------------------------------------
def build_path_lookup():
    df = pd.read_csv(METADATA_CSV)
    # filename is unique per project (used throughout)
    return dict(zip(df["filename"], df["image_path"]))

# ------------------------------------------------------------
# 2. Draw one cell (thumbnail + label)
# ------------------------------------------------------------
def draw_cell(sheet, x0, y0, image_path, suspect_id):
    # Read image, resize, and paste
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.full((THUMB_SIZE, THUMB_SIZE), 240, dtype=np.uint8)
        cv2.putText(img, "MISSING", (10, THUMB_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    else:
        img = cv2.resize(img, (THUMB_SIZE, THUMB_SIZE),
                         interpolation=cv2.INTER_AREA)
        # Convert grayscale to BGR for pasting into a BGR sheet
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Paste thumbnail
    sheet[y0:y0 + THUMB_SIZE, x0:x0 + THUMB_SIZE] = img

    # Draw a thin border around the thumbnail
    cv2.rectangle(sheet, (x0, y0),
                  (x0 + THUMB_SIZE, y0 + THUMB_SIZE),
                  (60, 60, 60), 1)

    # Draw label below thumbnail
    label = suspect_id
    label_y = y0 + THUMB_SIZE + 18
    cv2.putText(sheet, label, (x0 + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

# ------------------------------------------------------------
# 3. Build one sheet
# ------------------------------------------------------------
def build_sheet(shape_name, batch_number, batch_rows, path_lookup, out_path):
    sheet = np.full((SHEET_HEIGHT, SHEET_WIDTH, 3), 255, dtype=np.uint8)

    # Header
    title = f"{shape_name} — Batch {batch_number} of {len(batch_rows)}"
    cv2.putText(sheet, title, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Cells
    for idx, row in enumerate(batch_rows):
        col = idx % COLS
        row_idx = idx // COLS
        x0 = col * (THUMB_SIZE + 2 * CELL_PADDING) + CELL_PADDING
        y0 = HEADER_HEIGHT + row_idx * CELL_HEIGHT + CELL_PADDING
        image_path = path_lookup.get(row["filename"], "")
        draw_cell(sheet, x0, y0, image_path, row["suspect_id"])

    # Footer — list of SUS IDs on this sheet
    ids = " | ".join(r["suspect_id"] for r in batch_rows)
    cv2.putText(sheet, ids, (12, SHEET_HEIGHT - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, sheet)
    print(f"  Saved: {os.path.basename(out_path)}  ({len(batch_rows)} suspects)")

# ------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading suspects and path lookup...")
    suspects = pd.read_csv(SUSPECTS_CSV)
    path_lookup = build_path_lookup()
    print(f"Total suspects: {len(suspects)}")

    # Group by shape, preserving alphabetical order
    order = ["Circle", "Diamond", "Overlapped Circles", "Overlapped Pencils",
             "Square", "Star", "Triangle", "Wave"]

    sheets_created = 0
    for shape in order:
        group = suspects[suspects["logical_shape"] == shape].reset_index(drop=True)
        if len(group) == 0:
            continue
        n_batches = math.ceil(len(group) / BATCH_SIZE)
        print(f"\n=== {shape}: {len(group)} suspects → {n_batches} sheet(s) ===")
        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(group))
            batch_rows = group.iloc[start:end].to_dict("records")
            safe_shape = shape.replace(" ", "_")
            out_path = os.path.join(
                OUTPUT_DIR, f"sheet_{safe_shape}_{b + 1}.png")
            build_sheet(shape, b + 1, batch_rows, path_lookup, out_path)
            sheets_created += 1

    print("\n" + "=" * 60)
    print(f"DONE. {sheets_created} contact sheets saved to:")
    print(OUTPUT_DIR)
    print("=" * 60)