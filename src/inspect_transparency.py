from pathlib import Path
from PIL import Image
from collections import Counter

# 1. Dataset location
DATASET_ROOT = Path(r"C:\Users\dell\Downloads\Shapes\Shapes")

# 2. Counters
rgba_images = 0
fully_transparent_images = 0
partially_transparent_images = 0
no_transparency_images = 0
unreadable_images = 0

# 3. Inspect each shape folder
for shape_folder in DATASET_ROOT.iterdir():

    if not shape_folder.is_dir():
        continue

    for image_path in shape_folder.glob("*.png"):

        try:
            with Image.open(image_path) as image:

                if image.mode != "RGBA":
                    continue

                rgba_images += 1

                # Extract transparency information
                alpha = image.getchannel("A")

                alpha_counts = Counter(alpha.getdata())

                transparent_pixels = alpha_counts.get(0, 0)
                opaque_pixels = alpha_counts.get(255, 0)

                total_pixels = image.width * image.height

                partially_transparent_pixels = (
                    total_pixels
                    - transparent_pixels
                    - opaque_pixels
                )

                if transparent_pixels > 0:
                    fully_transparent_images += 1

                if partially_transparent_pixels > 0:
                    partially_transparent_images += 1

                if transparent_pixels == 0 and partially_transparent_pixels == 0:
                    no_transparency_images += 1

        except Exception:
            unreadable_images += 1

# 4. Display aggregate results
print("\nTRANSPARENCY INSPECTION")
print("-----------------------------")

print("RGBA images:", rgba_images)
print("Images with fully transparent pixels:", fully_transparent_images)
print("Images with partially transparent pixels:", partially_transparent_images)
print("Images without transparent pixels:", no_transparency_images)
print("Unreadable images:", unreadable_images)

print("\nInspection completed.")
print("Original images were not modified.")