import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ======================================================
# 1. Load Dataset
# ======================================================
input_path = "C:/Users/asus/Desktop/FYP Improvement/FYP2/cardio_train.csv"  # Your original dataset (semicolon CSV)

df = pd.read_csv(input_path, sep=";")

print("Original dataset shape:", df.shape)

# ======================================================
# 2. Convert age (days) -> age_years
# ======================================================
# No removal of data, only conversion
df["age_years"] = df["age"] / 365.0

# ======================================================
# 3. Select Columns (Optional: keep full dataset)
# ======================================================
# Keep all original columns + age_years
# No removal of height/weight rules
df_clean = df.copy()

# ======================================================
# 4. Train / Validation / Test Split (70/15/15)
# ======================================================
# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.30,
    random_state=42,
    stratify=df_clean["cardio"]  # maintain class balance
)

# Second split: from 30% temp → 15/15
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["cardio"]
)

print("Train set:", train_df.shape)
print("Validation set:", val_df.shape)
print("Test set:", test_df.shape)

# ======================================================
# 5. Save all into one folder
# ======================================================
output_dir = "C:/Users/asus/Desktop/FYP Improvement/FYP2/split_dataset"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("/nSaved all files to:", output_dir)
print("Files created:")
print(" - split_dataset/train.csv")
print(" - split_dataset/val.csv")
print(" - split_dataset/test.csv")
