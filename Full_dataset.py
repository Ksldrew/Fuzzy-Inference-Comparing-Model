"""
prepare_dataset.py

功能：
1. 读取已清洗的数据集（age 已删除，只保留 age_years）
2. 不进行年龄转换
3. 按 70% / 15% / 15% 分层划分 train / val / test
4. 保存四个文件
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ===========================
# 1. 配置
# ===========================
INPUT_PATH = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\full_dataset_final_cleaned_preprocessing.csv"
OUTPUT_DIR = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Preprocessing_Edited_Dataset"
TARGET_COL = "cardio"


def main():
    # ===========================
    # 2. 读取数据（关键修复点）
    # ===========================
    df = pd.read_csv(INPUT_PATH)
    print("Original dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # ===========================
    # 3. 校验
    # ===========================
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    if "age" in df.columns:
        raise ValueError("Column 'age' still exists. Please remove it.")

    if "age_years" not in df.columns:
        raise ValueError("Column 'age_years' not found.")

    df_clean = df.copy()

    # ===========================
    # 4. 分层切分（70 / 15 / 15）
    # ===========================
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=0.30,
        random_state=42,
        stratify=df_clean[TARGET_COL]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[TARGET_COL]
    )

    print("Train shape:", train_df.shape)
    print("Val shape  :", val_df.shape)
    print("Test shape :", test_df.shape)

    # ===========================
    # 5. 保存
    # ===========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_clean.to_csv(os.path.join(OUTPUT_DIR, "full_dataset.csv"), index=False)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("\nSaved all files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
