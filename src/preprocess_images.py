import os
import pandas as pd
import cv2

# 1. Define the paths
metadata_path = r"C:\Users\dell\Documents\ShapeScoringProject\outputs\tables\dataset_metadata.csv"
processed_folder = r"C:\Users\dell\Documents\ShapeScoringProject\outputs\processed_images"

# Create the processed folder if it doesn't exist
os.makedirs(processed_folder, exist_ok=True)

# 2. Load the metadata
df = pd.read_csv(metadata_path)

print(f"Starting preprocessing for {len(df)} images...")
print("This may take a minute or two on a CPU. Please wait.\n")

success_count = 0
error_count = 0

# Target size for all images
TARGET_SIZE = (128, 128)

# 3. Loop through each image in our metadata
for index, row in df.iterrows():
    # Print progress every 500 images so you know it's working
    if index % 500 == 0:
        print(f"Processing image {index + 1} of {len(df)}...")
        
    original_path = row['image_path']
    filename = row['filename']
    save_path = os.path.join(processed_folder, filename)
    
    try:
        # Read the image in grayscale (cv2.IMREAD_GRAYSCALE forces it to black and white)
        img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image was actually read
        if img is None:
            print(f"Warning: Could not read {filename}")
            error_count += 1
            continue
            
        # Resize the image to 128x128
        # cv2.INTER_AREA is the best mathematical method for shrinking images
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Save the processed image to the new folder
        cv2.imwrite(save_path, img_resized)
        success_count += 1
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        error_count += 1

print("\n--- Preprocessing Complete ---")
print(f"Successfully processed: {success_count} images.")
print(f"Errors: {error_count} images.")
print(f"Processed images saved in: {processed_folder}")