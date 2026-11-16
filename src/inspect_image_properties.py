from pathlib import Path
from PIL import Image
from collections import Counter

# 1. Dataset location
DATASET_ROOT = Path(r"C:\Users\dell\Downloads\Shapes\Shapes")

# 2. Store aggregate information
total_images = 0
unreadable_images = 0
transparent_images = 0

dimension_counts = Counter()
mode_counts = Counter()

min_width = float("inf")
max_width = 0
min_height = float("inf")
max_height = 0

# 3. Read each shape folder
for shape_folder in sorted(DATASET_ROOT.iterdir()):

    if not shape_folder.is_dir():
        continue

    folder_count = 0

    for image_path in shape_folder.glob("*.png"):

        total_images += 1
        folder_count += 1

        try:
            # Open image without modifying it
            with Image.open(image_path) as image:

                width, height = image.size

                dimension_counts[(width, height)] += 1
                mode_counts[image.mode] += 1

                min_width = min(min_width, width)
                max_width = max(max_width, width)

                min_height = min(min_height, height)
                max_height = max(max_height, height)

                # Check transparency
                has_transparency = (
                    "transparency" in image.info
                    or "A" in image.getbands()
                )

                if has_transparency:
                    transparent_images += 1

        except Exception:
            unreadable_images += 1

    print(f"{shape_folder.name}: {folder_count} images inspected")

# 4. Display aggregate results
print("\nIMAGE PROPERTY INSPECTION")
print("------------------------------")

print("Total images:", total_images)
print("Unreadable images:", unreadable_images)
print("Images with transparency:", transparent_images)

print("\nIMAGE DIMENSIONS")
print("Minimum width:", min_width)
print("Maximum width:", max_width)
print("Minimum height:", min_height)
print("Maximum height:", max_height)

print("\nIMAGE COLOUR MODES")
for mode, count in sorted(mode_counts.items()):
    print(mode, ":", count)

print("\nMOST COMMON DIMENSIONS")
for dimensions, count in dimension_counts.most_common(10):
    print(dimensions, ":", count)

print("\nInspection completed.")
print("Original images were not modified.")