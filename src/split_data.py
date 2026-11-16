import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# 1. Define the paths
metadata_path = r"C:\Users\dell\Documents\ShapeScoringProject\outputs\tables\dataset_metadata.csv"
train_path = r"C:\Users\dell\Documents\ShapeScoringProject\outputs\tables\train_metadata.csv"
test_path = r"C:\Users\dell\Documents\ShapeScoringProject\outputs\tables\test_metadata.csv"

# 2. Load the metadata
df = pd.read_csv(metadata_path)
# Drop any image that doesn't have a valid score
df = df.dropna(subset=['score']) 
print(f"Loaded {len(df)} valid images from metadata.\n")

# 3. Create a combined label for stratification
df['stratify_label'] = df['logical_shape'] + "_" + df['score'].astype(int).astype(str)

# 4. Handle ultra-rare classes (THE FIX)
# Stratified split needs at least 2 images per class. 
# We find classes with only 1 image and remove them.
class_counts = df['stratify_label'].value_counts()
rare_classes = class_counts[class_counts < 2].index
print(f"Removing {len(rare_classes)} ultra-rare classes with only 1 image: {list(rare_classes)}")
df = df[~df['stratify_label'].isin(rare_classes)]
print(f"Remaining images for splitting: {len(df)}\n")

# 5. Perform the Stratified Split (80% Train, 20% Test)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(sss.split(df, df['stratify_label']))

df_train = df.iloc[train_indices].copy()
df_test = df.iloc[test_indices].copy()

# 6. Save the splits to CSV
df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print("--- Data Splitting Complete (Stratified) ---")
print(f"Training set size: {len(df_train)} images ({len(df_train)/len(df)*100:.1f}%)")
print(f"Testing set size:  {len(df_test)} images ({len(df_test)/len(df)*100:.1f}%)\n")

# 7. Verify the split is fair
print("--- Verification: Score Distribution by Shape ---")
print("Notice how the scores are now balanced in the Test set!\n")

for shape in df['logical_shape'].unique():
    print(f"Shape: {shape}")
    
    train_counts = df_train[df_train['logical_shape'] == shape]['score'].value_counts().sort_index()
    train_total = len(df_train[df_train['logical_shape'] == shape])
    
    test_counts = df_test[df_test['logical_shape'] == shape]['score'].value_counts().sort_index()
    test_total = len(df_test[df_test['logical_shape'] == shape])
    
    print(f"  Train ({train_total} total): Scores -> {dict(train_counts)}")
    print(f"  Test  ({test_total} total): Scores -> {dict(test_counts)}")
    print("-" * 40)

print("\nFiles saved successfully!")
print(f"Train: {train_path}")
print(f"Test:  {test_path}")