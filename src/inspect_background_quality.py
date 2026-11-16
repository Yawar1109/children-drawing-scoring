
from pathlib import Path
from PIL import Image
import numpy as np
import random
import csv

DATASET_ROOT = Path(r"C:\Users\dell\Downloads\Shapes\Shapes")

OUTPUT_FILE = Path(
    "outputs/tables/background_quality.csv"
)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)
SAMPLE_SIZE = 10

results = []

print("\nBACKGROUND QUALITY INSPECTION")
print("--------------------------------")

for shape_folder in sorted(DATASET_ROOT.iterdir()):

    if not shape_folder.is_dir():
        continue

    image_files = list(shape_folder.glob("*.png"))

    selected = random.sample(
        image_files,
        min(SAMPLE_SIZE, len(image_files))
    )

    image_brightness = []
    image_variation = []

    for image_path in selected:

        try:
            with Image.open(image_path) as image:

                image = image.convert("RGB")
                image.thumbnail((300, 300))

                pixels = np.array(image)

                height, width, _ = pixels.shape

                # Select a border 10% wide
                border_y = max(1, int(height * 0.10))
                border_x = max(1, int(width * 0.10))

                border_pixels = np.concatenate([
                    pixels[:border_y, :, :].reshape(-1, 3),
                    pixels[-border_y:, :, :].reshape(-1, 3),
                    pixels[:, :border_x, :].reshape(-1, 3),
                    pixels[:, -border_x:, :].reshape(-1, 3)
                ])

                # Measure border brightness
                brightness = border_pixels.mean(axis=1)

                median = np.median(brightness)
                variation = np.std(brightness)

                image_brightness.append(median)
                image_variation.append(variation)

        except Exception:
            continue

    if image_brightness:

        results.append({
            "Shape": shape_folder.name,
            "Images_Sampled": len(image_brightness),
            "Median_Border_Brightness": round(
                np.mean(image_brightness), 2
            ),
            "Average_Border_Variation": round(
                np.mean(image_variation), 2
            )
        })

        print(f"\nShape: {shape_folder.name}")
        print("Images sampled:", len(image_brightness))
        print(
            "Median border brightness:",
            round(np.mean(image_brightness), 2)
        )
        print(
            "Average border variation:",
            round(np.mean(image_variation), 2)
        )

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fields = [
        "Shape",
        "Images_Sampled",
        "Median_Border_Brightness",
        "Average_Border_Variation"
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("\n--------------------------------")
print("Inspection completed.")
print("Original images were not modified.")
print("Results saved to:", OUTPUT_FILE)
