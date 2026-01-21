import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os

# === Load dataset (use train.csv to avoid data leakage) ===
df = pd.read_csv(r"C:/Users/asus/Desktop/FYP Improvement/FYP2/Edited_Dataset/train.csv")

target = "cardio"
feature_cols = [c for c in df.columns if c != target]

# === Pearson correlation ===
pearson = {}
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        pearson[c] = df[c].corr(df[target])

pearson_s = pd.Series(pearson).sort_values(key=lambda s: s.abs(), ascending=False)

# === Spearman correlation ===
spearman = {}
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        spearman[c] = spearmanr(df[c], df[target]).correlation

spearman_s = pd.Series(spearman).sort_values(key=lambda s: s.abs(), ascending=False)

print("\nTop Pearson correlations:\n", pearson_s.head(10))
print("\nTop Spearman correlations:\n", spearman_s.head(10))

# === Save directory ===
save_dir = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\correlation"
os.makedirs(save_dir, exist_ok=True)

# === Save CSV ===
out = pd.DataFrame({
    "pearson_r": pearson_s,
    "spearman_rho": spearman_s.reindex(pearson_s.index)
}).reset_index().rename(columns={"index": "feature"})

csv_path = os.path.join(save_dir, "feature_correlation_with_cardio.csv")
out.to_csv(csv_path, index=False)

# === Plot Pearson correlation (Top 10) ===
plt.figure(figsize=(8, 5))
pearson_s.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("Pearson Correlation")
plt.title("Top 10 Pearson Correlation with CVD")
plt.tight_layout()

pearson_png = os.path.join(save_dir, "pearson_correlation.png")
plt.savefig(pearson_png, dpi=300)
plt.close()

# === Plot Spearman correlation (Top 10) ===
plt.figure(figsize=(8, 5))
spearman_s.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("Spearman Correlation")
plt.title("Top 10 Spearman Correlation with CVD")
plt.tight_layout()

spearman_png = os.path.join(save_dir, "spearman_correlation.png")
plt.savefig(spearman_png, dpi=300)
plt.close()

print("\nSaved files:")
print(csv_path)
print(pearson_png)
print(spearman_png)
