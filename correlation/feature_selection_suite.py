"""
Feature Selection Suite for CVD (cardio: 0/1)
Methods included:
- Pearson correlation (linear)
- Spearman correlation (monotonic)
- Point-biserial correlation (continuous feature vs binary target)
- Mutual Information (nonlinear dependency; classification)
Outputs:
- CSV ranking per method
- Top-K barh PNG per method
- Optional correlation heatmap PNG (Pearson on numeric features)

Author: ChatGPT (comments in English as requested)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_classif

# =========================
# User config
# =========================
DATA_PATH = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\train.csv"
TARGET_COL = "cardio"

# Save everything under this folder (each method has its own subfolder)
BASE_OUT_DIR = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\correlation\new folder"

TOP_K = 15                 # how many features to show in bar plot
RANDOM_STATE = 42          # for MI
MI_DISCRETE_AUTO = "auto"  # let sklearn infer discrete features (works for mixed types)

# =========================
# Helpers
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def numeric_feature_cols(df: pd.DataFrame, target: str) -> list:
    cols = []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def save_ranking_and_plot(method_name: str, scores: pd.Series, base_out_dir: str, top_k: int = 15):
    """
    Save full ranking CSV + Top-K barh plot PNG for one method.
    """
    out_dir = ensure_dir(os.path.join(base_out_dir, method_name))
    scores = scores.dropna()

    # Save full ranking
    csv_path = os.path.join(out_dir, f"{method_name}_ranking.csv")
    scores.to_frame("score").reset_index().rename(columns={"index": "feature"}).to_csv(csv_path, index=False)

    # Plot Top-K by absolute score (for correlation-like metrics)
    top = scores.reindex(scores.abs().sort_values(ascending=False).head(top_k).index)

    plt.figure(figsize=(9, max(4, 0.35 * len(top))))
    top.sort_values().plot(kind="barh")  # sort for nicer barh view
    plt.xlabel("Score")
    plt.title(f"Top {min(top_k, len(scores))} Features by {method_name}")
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"{method_name}_top{min(top_k, len(scores))}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    return csv_path, png_path

def save_heatmap(df: pd.DataFrame, cols: list, out_dir: str, filename: str = "pearson_heatmap.png"):
    """
    Save a Pearson correlation heatmap for numeric features (including target).
    Uses matplotlib only (no seaborn), per your environment preferences.
    """
    if len(cols) < 2:
        return None

    mat = df[cols].corr(method="pearson").values

    plt.figure(figsize=(0.45 * len(cols) + 4, 0.45 * len(cols) + 4))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Pearson Correlation Heatmap (Numeric Features)")
    plt.tight_layout()

    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    return path

# =========================
# Main
# =========================
def main():
    ensure_dir(BASE_OUT_DIR)

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    # Keep only rows with non-missing target
    df = df.dropna(subset=[TARGET_COL]).copy()

    # Identify numeric features
    num_cols = numeric_feature_cols(df, TARGET_COL)

    # Ensure target is numeric 0/1
    y = df[TARGET_COL]
    if not pd.api.types.is_numeric_dtype(y):
        # Try to map common labels to 0/1
        y = y.astype(str).str.lower().map({"0": 0, "1": 1, "false": 0, "true": 1})
    y = y.astype(float)

    # ========= Pearson =========
    pearson_scores = {}
    for c in num_cols:
        pearson_scores[c] = df[c].corr(y)
    pearson_s = pd.Series(pearson_scores, name="pearson").sort_values(key=lambda s: s.abs(), ascending=False)
    pearson_csv, pearson_png = save_ranking_and_plot("pearson", pearson_s, BASE_OUT_DIR, TOP_K)

    # Heatmap (optional but useful for documentation)
    heatmap_dir = ensure_dir(os.path.join(BASE_OUT_DIR, "pearson_heatmap"))
    heatmap_cols = [TARGET_COL] + num_cols
    heatmap_path = save_heatmap(df.assign(**{TARGET_COL: y}), heatmap_cols, heatmap_dir)

    # ========= Spearman =========
    spearman_scores = {}
    for c in num_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman_scores[c] = spearmanr(df[c], y, nan_policy="omit").correlation
    spearman_s = pd.Series(spearman_scores, name="spearman").sort_values(key=lambda s: s.abs(), ascending=False)
    spearman_csv, spearman_png = save_ranking_and_plot("spearman", spearman_s, BASE_OUT_DIR, TOP_K)

    # ========= Point-Biserial =========
    # Best when feature is continuous and target is binary.
    pb_scores = {}
    for c in num_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pb_scores[c] = pointbiserialr(df[c], y).correlation
    pb_s = pd.Series(pb_scores, name="point_biserial").sort_values(key=lambda s: s.abs(), ascending=False)
    pb_csv, pb_png = save_ranking_and_plot("point_biserial", pb_s, BASE_OUT_DIR, TOP_K)

    # ========= Mutual Information =========
    # MI can capture nonlinear relationships; it returns non-negative scores.
    # For mixed types, keep numeric cols here (simple + stable). You can extend later.
    X = df[num_cols].copy()

    # Fill missing values (MI cannot handle NaNs)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    mi = mutual_info_classif(
        X.values,
        y.values.astype(int),
        discrete_features=MI_DISCRETE_AUTO,
        random_state=RANDOM_STATE
    )
    mi_s = pd.Series(mi, index=num_cols, name="mutual_info").sort_values(ascending=False)
    mi_csv, mi_png = save_ranking_and_plot("mutual_information", mi_s, BASE_OUT_DIR, TOP_K)

    # ========= Combined summary (one CSV for report) =========
    summary_dir = ensure_dir(os.path.join(BASE_OUT_DIR, "summary"))
    summary = pd.DataFrame({
        "pearson_r": pearson_s,
        "spearman_rho": spearman_s,
        "point_biserial_r": pb_s,
        "mutual_info": mi_s
    }).reset_index().rename(columns={"index": "feature"})

    summary_path = os.path.join(summary_dir, "feature_selection_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Print outputs
    print("\nSaved outputs:")
    print("Pearson:", pearson_csv, pearson_png)
    if heatmap_path:
        print("Pearson heatmap:", heatmap_path)
    print("Spearman:", spearman_csv, spearman_png)
    print("Point-biserial:", pb_csv, pb_png)
    print("Mutual Information:", mi_csv, mi_png)
    print("Summary:", summary_path)

if __name__ == "__main__":
    main()
