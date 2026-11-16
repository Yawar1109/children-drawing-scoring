import os
import re
import pandas as pd

# 1. Define the paths
dataset_path = r"C:\Users\dell\Downloads\Shapes\Shapes"
output_csv_path = r"C:\Users\dell\Documents\ShapeScoringProject\outputs\tables\dataset_metadata.csv"

# 2. Map folder names to logical shape names
folder_to_shape = {
    "Circle": "Circle",
    "Diagonal": "Diamond",
    "Overlapped circle": "Overlapped Circles",
    "Overlapped pencils": "Overlapped Pencils",
    "Square": "Square",
    "Star": "Star",
    "Triangle": "Triangle",
    "Wave": "Wave"
}

# 3. Prepare empty lists to hold our data
metadata = []
anomalies = []

# 4. Loop through each folder in the dataset
for folder_name, logical_shape in folder_to_shape.items():
    folder_path = os.path.join(dataset_path, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found!")
        continue
        
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.png'):
            continue
            
        image_path = os.path.join(folder_path, filename)
        base_name = filename[:-4]
        
        # Look for suffix in parentheses like (123)
        suffix_match = re.search(r'\((\d+)\)', base_name)
        suffix_value = None
        if suffix_match:
            suffix_value = int(suffix_match.group(1))
            base_name_without_suffix = base_name[:suffix_match.start()] + base_name[suffix_match.end():]
        else:
            base_name_without_suffix = base_name
            
        # Extract the score (the last number in the filename)
        numbers = re.findall(r'\d+', base_name_without_suffix)
        
        score = None
        if numbers:
            score = int(numbers[-1])
        else:
            anomalies.append({
                "folder": folder_name,
                "filename": filename,
                "reason": "No numbers found in filename"
            })
            
        if logical_shape == "Star" and score == 6:
            anomalies.append({
                "folder": folder_name,
                "filename": filename,
                "reason": "Star image has score 6 (max is 5)"
            })

        metadata.append({
            "image_path": image_path,
            "folder_name": folder_name,
            "logical_shape": logical_shape,
            "filename": filename,
            "score": score,
            "has_suffix": suffix_value is not None,
            "suffix_value": suffix_value
        })

# 5. Convert to Pandas DataFrames and save to CSV
df_metadata = pd.DataFrame(metadata)
df_anomalies = pd.DataFrame(anomalies)

df_metadata.to_csv(output_csv_path, index=False)
print(f"Successfully saved metadata for {len(df_metadata)} images to:\n{output_csv_path}\n")

if not df_anomalies.empty:
    anomalies_path = r"C:\Users\dell\Documents\ShapeScoringProject\outputs\tables\dataset_anomalies.csv"
    df_anomalies.to_csv(anomalies_path, index=False)
    print(f"Found {len(df_anomalies)} anomalies. Saved to:\n{anomalies_path}\n")
else:
    print("No anomalies found during parsing.\n")

print("--- Quick Summary ---")
print(f"Total images processed: {len(df_metadata)}")
print("\nImages per shape:")
print(df_metadata['logical_shape'].value_counts().to_string())

print("\n--- Verification: First 3 filenames per shape ---")
for shape in df_metadata['logical_shape'].unique():
    shape_files = df_metadata[df_metadata['logical_shape'] == shape]['filename'].head(3).tolist()
    print(f"{shape}: {shape_files}")