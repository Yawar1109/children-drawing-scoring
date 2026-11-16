
from pathlib import Path
from PIL import Image
import numpy as np
import random
import csv

# 1. Dataset location
DATASET_ROOT = Path(r"C:\Users\dell\Downloads\Shapes\Shapes")

# 2. Output location
OUTPUT_FILE = Path("outputs/tables/background_inspection.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 3. Select a reproducible random sample
random.seed(42)
SAMPLE_SIZE = 10

results = []

print("\nFULL IMAGE BACKGROUND INSPECTION")
print("--------------------------------")

for shape_folder in sorted(DATASET_ROOT.iterdir()):

    if not shape_folder.is_dir():
        continue

    image_files = list(shape_folder.glob("*.png"))

    selected_images = random.sample(
        image_files,
        min(SAMPLE_SIZE, len(image_files))
    )

    brightness_values = []
    white_pixel_percentages = []

    for image_path in selected_images:

        try:
            with Image.open(image_path) as image:

                # Convert a temporary copy to RGB
                rgb_image = image.convert("RGB")

                # Resize only the temporary copy for faster inspection
                rgb_image.thumbnail((300, 300))

                pixels = np.array(rgb_image)

                # Calculate brightness of every pixel
                brightness = pixels.mean(axis=2)

                # Calculate median brightness
                median_brightness = np.median(brightness)

                # Count near-white pixels
                near_white = np.all(pixels >= 240, axis=2)

                white_percentage = (
                    np.mean(near_white) * 100
                )

                brightness_values.append(median_brightness)
                white_pixel_percentages.append(white_percentage)

        except Exception:
            continue

    if brightness_values:

        average_median = np.mean(brightness_values)
        average_white = np.mean(white_pixel_percentages)

        results.append({
            "Shape": shape_folder.name,
            "Images_Sampled": len(brightness_values),
            "Average_Median_Brightness": round(
                average_median, 2
            ),
            "Average_Near_White_Pixels_Percent": round(
                average_white, 2
            )
        })

        print(f"\nShape: {shape_folder.name}")
        print("Images sampled:", len(brightness_values))
        print(
            "Average median brightness:",
            round(average_median, 2)
        )
        print(
            "Average near-white pixels (%):",
            round(average_white, 2)
        )

# 4. Save results to CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:

    fieldnames = [
        "Shape",
        "Images_Sampled",
        "Average_Median_Brightness",
        "Average_Near_White_Pixels_Percent"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

print("\n--------------------------------")
print("Inspection completed.")
print("Original images were not modified.")
print("Results saved to:", OUTPUT_FILE)
