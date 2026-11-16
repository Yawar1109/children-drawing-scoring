from pathlib import Path
import re
import csv

# 1. Location of the original dataset
DATASET_ROOT = Path(r"C:\Users\dell\Downloads\Shapes\Shapes")

# 2. Location where summary files will be saved
OUTPUT_FOLDER = Path("outputs") / "tables"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 3. Pattern for reading image scores from filenames
filename_pattern = re.compile(
    r"^img\d+-[A-Za-z]+-(\d+)(?:\(\d+\))?$"
)

# 4. Store aggregate results only
summary_rows = []
score_rows = []

total_images = 0
total_with_parentheses = 0
total_unparsed = 0

# 5. Check that the dataset folder exists
if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        "Dataset folder not found. Please check the dataset location."
    )

# 6. Read each shape folder
for shape_folder in sorted(DATASET_ROOT.iterdir()):

    if not shape_folder.is_dir():
        continue

    image_files = [
        file for file in shape_folder.iterdir()
        if file.is_file() and file.suffix.lower() == ".png"
    ]

    score_counts = {}
    parenthesis_count = 0
    unparsed_count = 0

    # 7. Inspect filenames without displaying them
    for image_file in image_files:

        filename_without_extension = image_file.stem

        if re.search(r"\(\d+\)$", filename_without_extension):
            parenthesis_count += 1

        match = filename_pattern.fullmatch(
            filename_without_extension
        )

        if match:
            score = int(match.group(1))
            score_counts[score] = score_counts.get(score, 0) + 1
        else:
            unparsed_count += 1

    # 8. Save one summary row for each shape
    summary_rows.append({
        "Shape folder": shape_folder.name,
        "Total PNG images": len(image_files),
        "Filenames with parentheses": parenthesis_count,
        "Filenames not parsed": unparsed_count
    })

    # 9. Save score counts separately
    for score, count in sorted(score_counts.items()):
        score_rows.append({
            "Shape folder": shape_folder.name,
            "Expert score": score,
            "Number of images": count
        })

    total_images += len(image_files)
    total_with_parentheses += parenthesis_count
    total_unparsed += unparsed_count

# 10. Save aggregate summary CSV
summary_path = OUTPUT_FOLDER / "dataset_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Shape folder",
            "Total PNG images",
            "Filenames with parentheses",
            "Filenames not parsed"
        ]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

# 11. Save score distribution CSV
scores_path = OUTPUT_FOLDER / "score_distribution.csv"

with scores_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Shape folder",
            "Expert score",
            "Number of images"
        ]
    )
    writer.writeheader()
    writer.writerows(score_rows)

# 12. Display only safe aggregate information
print("\nDATASET INSPECTION COMPLETED")
print("--------------------------------")
print("Total PNG images:", total_images)
print("Shape folders:", len(summary_rows))
print("Filenames with parentheses:", total_with_parentheses)
print("Filenames not parsed:", total_unparsed)

print("\nImages in each folder:")
for row in summary_rows:
    print(row["Shape folder"], ":", row["Total PNG images"])

print("\nSummary files saved:")
print(summary_path)
print(scores_path)

print("\nOriginal dataset was not modified.")