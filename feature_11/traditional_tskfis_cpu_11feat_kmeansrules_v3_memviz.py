# -*- coding: utf-8 -*-
"""
Traditional (Monolithic) TSK-FIS (CPU) — 11 Features, KMeans Rule Reduction (NO full grid)
======================================================================================

This script is designed for your Chapter 4 reporting:
- It runs a *traditional monolithic* TSK-FIS pipeline (firing -> normalization -> weighted sum).
- It avoids *grid-partition rule explosion* by generating a fixed number of rules K using KMeans.
- It exports ALL visualizations to disk (no interactive show).
- It writes a memory feasibility note and runtime memory logs so you can cite "32GB RAM not enough"
  if you switch to full grid-partition (hundreds of thousands of rules).

Key additions vs earlier versions:
1) Memory-aware inference (no huge NxR "rule_out" unless needed).
2) Uses float32 for large matrices to reduce RAM.
3) Saves 3D surface figures for the same (A,B) pairs used in 2D slice plots INTO the slice folder.
4) Exports memory estimation + runtime RSS/peak stats into exports/memory_report.json.

Comments are in English (as you requested for code).
"""

import os
import json
import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional: better runtime memory reporting
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import tracemalloc


# ==============================
# 0) USER SETTINGS (EDIT HERE)
# ==============================

# Option A (recommended): already-split CSVs
TRAIN_CSV = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\train.csv"
VAL_CSV   = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\val.csv"
TEST_CSV  = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\test.csv"

# Option B: not used when train/val/test are provided
FULL_CSV  = ""


TARGET_COL = "cardio"

# --- Features (11) ---
FEATURE_COLS = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo",
    "cholesterol", "gluc",
    "smoke", "alco", "active"
]

# --- Output ---
OUT_ROOT  = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\feature_11\without_grid"
RUN_NAME  = "traditional_tskfis_cpu_11feat_kmeansrules"

# --- Preprocessing ---
USE_ZSCORE = True
RANDOM_STATE = 42

# --- Rule reduction (critical) ---
NUM_RULES = 120            # K rules (typical 50~200). Increase carefully.
SIGMA_FLOOR = 1e-3         # prevents divide-by-zero

# --- Consequent training ---
RIDGE_ALPHA = 1.0          # increase to regularize more

# --- Probability mapping / classification ---
USE_SIGMOID = True
THRESHOLD = 0.5

# --- Visualization controls ---
GRID_2D = 250
GRID_3D = 45
SLICE_LEVELS = [0.25, 0.50, 0.75]
MAX_3D_PAIRS = 12
MAX_2D_SLICE_PAIRS = 12

# --- Explain export ---
EXPLAIN_N = 10
TOP_RULES_TO_SHOW = 12

# --- Memory control ---
RULE_CHUNK = 40            # smaller => lower RAM, slower
DTYPE = np.float32         # IMPORTANT: reduce memory usage vs float64

# --- Grid-partition feasibility note (for Chapter 4 reporting only) ---
# Keep USE_GRID_PARTITION=False for 11 features.
USE_GRID_PARTITION = False

MF_COUNTS = {
    "age_years": 5,
    "gender": 2,
    "height": 4,
    "weight": 4,
    "ap_hi": 5,
    "ap_lo": 5,
    "cholesterol": 3,
    "gluc": 3,
    "smoke": 2,
    "alco": 2,
    "active": 2,
}


# ==============================
# 1) HELPERS
# ==============================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def rss_gb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024**3)
    except Exception:
        return None

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def linspace_in_range(series: pd.Series, n: int) -> np.ndarray:
    lo = float(series.quantile(0.01))
    hi = float(series.quantile(0.99))
    if lo == hi:
        lo, hi = float(series.min()), float(series.max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, n)

def make_run_dirs(out_root: str, run_name: str) -> Dict[str, str]:
    base = ensure_dir(os.path.join(out_root, run_name))
    return {
        "base": base,
        "exports": ensure_dir(os.path.join(base, "exports")),
        "figures": ensure_dir(os.path.join(base, "figures")),
        "fig_2d": ensure_dir(os.path.join(base, "figures", "surfaces_2d")),
        "fig_2d_slices": ensure_dir(os.path.join(base, "figures", "surfaces_2d_slices")),
        "fig_3d": ensure_dir(os.path.join(base, "figures", "surfaces_3d")),
        "explain": ensure_dir(os.path.join(base, "step1to5_explain")),
    }

def _derive_age_years_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "age_years" not in df.columns and "age_days" in df.columns:
        df = df.copy()
        df["age_years"] = df["age_days"] / 365.25
    return df

def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).reset_index(drop=True)
    return df

def load_or_split_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if TRAIN_CSV and VAL_CSV and TEST_CSV:
        train_df = _basic_clean(_derive_age_years_if_needed(pd.read_csv(TRAIN_CSV)))
        val_df   = _basic_clean(_derive_age_years_if_needed(pd.read_csv(VAL_CSV)))
        test_df  = _basic_clean(_derive_age_years_if_needed(pd.read_csv(TEST_CSV)))

        for c in FEATURE_COLS + [TARGET_COL]:
            for name, d in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
                if c not in d.columns:
                    raise ValueError(f"Missing column '{c}' in {name}_CSV")

        X_train = train_df[FEATURE_COLS].copy()
        y_train = train_df[TARGET_COL].astype(int).copy()
        X_val   = val_df[FEATURE_COLS].copy()
        y_val   = val_df[TARGET_COL].astype(int).copy()
        X_test  = test_df[FEATURE_COLS].copy()
        y_test  = test_df[TARGET_COL].astype(int).copy()
        return X_train, y_train, X_val, y_val, X_test, y_test

    df = _basic_clean(_derive_age_years_if_needed(pd.read_csv(FULL_CSV)))
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in FULL_CSV: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp
    )
    return (X_train.reset_index(drop=True), y_train.reset_index(drop=True),
            X_val.reset_index(drop=True), y_val.reset_index(drop=True),
            X_test.reset_index(drop=True), y_test.reset_index(drop=True))

def estimate_grid_rules(feature_cols: List[str], mf_counts: Dict[str, int]) -> int:
    R = 1
    for c in feature_cols:
        if c not in mf_counts:
            raise ValueError(f"MF_COUNTS missing feature '{c}'")
        R *= int(mf_counts[c])
    return int(R)

def bytes_to_gb(x: float) -> float:
    return float(x) / (1024**3)

def estimate_tensor_gb(n: int, r: int, dtype_bytes: int) -> float:
    return bytes_to_gb(n * r * dtype_bytes)

@dataclass
class RuleAntecedent:
    mu: np.ndarray      # (F,)
    sigma: np.ndarray   # (F,)

def build_rules_kmeans(X_train_model: np.ndarray, K: int) -> List[RuleAntecedent]:
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_train_model)
    centers = km.cluster_centers_  # (K,F)

    rules: List[RuleAntecedent] = []
    for k in range(K):
        pts = X_train_model[labels == k]
        if pts.shape[0] <= 1:
            sigma = np.std(X_train_model, axis=0)
        else:
            sigma = np.std(pts, axis=0)
        sigma = np.maximum(sigma, SIGMA_FLOOR)
        rules.append(RuleAntecedent(mu=centers[k].astype(DTYPE), sigma=sigma.astype(DTYPE)))
    return rules

def compute_firing_chunked(X_model: np.ndarray, rules: List[RuleAntecedent], chunk: int = RULE_CHUNK) -> np.ndarray:
    """
    Returns firing matrix (N,R) in float32 to control RAM.
    """
    X_model = X_model.astype(DTYPE, copy=False)
    N, F = X_model.shape
    R = len(rules)
    firing = np.zeros((N, R), dtype=DTYPE)

    for start in range(0, R, chunk):
        end = min(start + chunk, R)
        mus = np.stack([rules[r].mu for r in range(start, end)], axis=0).astype(DTYPE, copy=False)   # (Rc,F)
        sig = np.stack([rules[r].sigma for r in range(start, end)], axis=0).astype(DTYPE, copy=False)# (Rc,F)

        x = X_model[:, None, :]  # (N,1,F)
        m = mus[None, :, :]      # (1,Rc,F)
        s = sig[None, :, :]      # (1,Rc,F)

        mf = np.exp(-0.5 * ((x - m) / s) ** 2, dtype=DTYPE)    # (N,Rc,F)
        firing[:, start:end] = np.prod(mf, axis=2, dtype=DTYPE) # (N,Rc)

    return firing

def normalize_firing(firing: np.ndarray) -> np.ndarray:
    denom = firing.sum(axis=1, keepdims=True) + DTYPE(1e-12)
    return firing / denom

def train_ridge_consequents(X_model: np.ndarray, y: np.ndarray, firing: np.ndarray) -> np.ndarray:
    """
    Consequent per rule: [p1..pF, bias] => shape (R, F+1).
    """
    X_model = X_model.astype(np.float64, copy=False)  # Ridge expects float64 for stability
    y = y.astype(np.float64, copy=False)

    N, F = X_model.shape
    R = firing.shape[1]
    X_ext = np.hstack([X_model, np.ones((N, 1), dtype=np.float64)])
    coef_mat = np.zeros((R, F + 1), dtype=np.float64)

    for r in range(R):
        w = firing[:, r].astype(np.float64, copy=False)
        if np.all(w < 1e-12):
            continue
        model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
        model.fit(X_ext, y, sample_weight=w)
        coef_mat[r, :] = model.coef_
        coef_mat[r, -1] += model.intercept_
    return coef_mat.astype(np.float32)

def infer_scores_light(X_model: np.ndarray, rules: List[RuleAntecedent], coef_mat: np.ndarray,
                       return_rule_mats: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Memory-aware inference:
    - Always returns score (N,).
    - If return_rule_mats=False: does NOT allocate NxR rule_out (large).
    - If True: returns firing, norm_firing, rule_out for explain/analysis.
    """
    firing = compute_firing_chunked(X_model, rules)
    nf = normalize_firing(firing)

    # Compute final score without storing full rule_out if not needed:
    # score = sum_r nf[:,r] * (X_ext @ coef_r)
    X_ext = np.hstack([X_model.astype(np.float32, copy=False), np.ones((X_model.shape[0], 1), dtype=np.float32)])
    coef_mat = coef_mat.astype(np.float32, copy=False)

    if not return_rule_mats:
        # Compute dot per rule in a streaming way to avoid NxR
        score = np.zeros((X_model.shape[0],), dtype=np.float32)
        # (N,F+1) dot (F+1,) per rule, then weighted sum
        for r in range(coef_mat.shape[0]):
            rule_out_r = (X_ext * coef_mat[r]).sum(axis=1)   # (N,)
            score += nf[:, r] * rule_out_r
        return score, {"firing": firing, "norm_firing": nf}

    # Full matrices (only for small n, e.g., explain)
    rule_out = X_ext @ coef_mat.T  # (N,R)
    score = (nf * rule_out).sum(axis=1)
    return score, {"firing": firing, "norm_firing": nf, "rule_out": rule_out}

# ==============================
# 2) VISUALIZATION
# ==============================
def plot_roc(y_true: np.ndarray, prob: np.ndarray, out_path: str):
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={auc:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_confusion(cm: np.ndarray, out_path: str):
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def response_prob_for_inputs(x_orig_row: np.ndarray,
                             scaler: Optional[StandardScaler],
                             rules: List[RuleAntecedent],
                             coef_mat: np.ndarray) -> float:
    x = x_orig_row.reshape(1, -1).astype(np.float32)
    x_model = scaler.transform(x).astype(np.float32) if scaler is not None else x
    score, _ = infer_scores_light(x_model, rules, coef_mat, return_rule_mats=False)
    if USE_SIGMOID:
        return float(sigmoid(score.astype(np.float64))[0])
    return float(score[0])

def plot_2d_single_curves(X_train_orig: pd.DataFrame,
                          scaler: Optional[StandardScaler],
                          rules: List[RuleAntecedent],
                          coef_mat: np.ndarray,
                          out_dir: str):
    med = X_train_orig[FEATURE_COLS].median().values.astype(np.float32)
    for i, feat in enumerate(FEATURE_COLS):
        xs = linspace_in_range(X_train_orig[feat], GRID_2D)
        ys = []
        for v in xs:
            x_row = med.copy()
            x_row[i] = np.float32(v)
            ys.append(response_prob_for_inputs(x_row, scaler, rules, coef_mat))
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(feat)
        plt.ylabel("Predicted probability" if USE_SIGMOID else "Predicted score")
        plt.title(f"2D Response Curve: {feat} (others fixed at median)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"2d_curve_{feat}.png"), dpi=300)
        plt.close()

def plot_3d_surface_for_pair(A: str, B: str,
                             X_train_orig: pd.DataFrame,
                             scaler: Optional[StandardScaler],
                             rules: List[RuleAntecedent],
                             coef_mat: np.ndarray,
                             out_path: str):
    med = X_train_orig[FEATURE_COLS].median().to_dict()
    a_idx = FEATURE_COLS.index(A)
    b_idx = FEATURE_COLS.index(B)

    xA = linspace_in_range(X_train_orig[A], GRID_3D)
    xB = linspace_in_range(X_train_orig[B], GRID_3D)
    XA, XB = np.meshgrid(xA, xB)
    Z = np.zeros_like(XA, dtype=np.float32)

    for i in range(XA.shape[0]):
        for j in range(XA.shape[1]):
            x_row = np.array([med[c] for c in FEATURE_COLS], dtype=np.float32)
            x_row[a_idx] = np.float32(XA[i, j])
            x_row[b_idx] = np.float32(XB[i, j])
            Z[i, j] = np.float32(response_prob_for_inputs(x_row, scaler, rules, coef_mat))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XA, XB, Z)
    ax.set_xlabel(A)
    ax.set_ylabel(B)
    ax.set_zlabel("Prob" if USE_SIGMOID else "Score")
    ax.set_title(f"3D Surface: {A} vs {B} (others=median)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_2d_slice_curves_with_surface(X_train_orig: pd.DataFrame,
                                      scaler: Optional[StandardScaler],
                                      rules: List[RuleAntecedent],
                                      coef_mat: np.ndarray,
                                      out_dir_slices: str,
                                      out_dir_surfaces: str,
                                      max_pairs: int = MAX_2D_SLICE_PAIRS,
                                      also_save_3d_in_slice_folder: bool = True):
    """
    Saves:
    - 2D slice curves in out_dir_slices
    - 3D surfaces:
        * always in out_dir_surfaces (limited)
        * additionally in out_dir_slices for the same pairs (if also_save_3d_in_slice_folder=True)
    """
    med = X_train_orig[FEATURE_COLS].median().to_dict()
    pairs = list(itertools.permutations(range(len(FEATURE_COLS)), 2))[:max_pairs]

    for a_idx, b_idx in pairs:
        A = FEATURE_COLS[a_idx]
        B = FEATURE_COLS[b_idx]
        xs = linspace_in_range(X_train_orig[A], GRID_2D)
        b_levels = [float(X_train_orig[B].quantile(q)) for q in SLICE_LEVELS]

        # --- 2D slice curves ---
        plt.figure()
        for lv in b_levels:
            ys = []
            for v in xs:
                x_row = np.array([med[c] for c in FEATURE_COLS], dtype=np.float32)
                x_row[a_idx] = np.float32(v)
                x_row[b_idx] = np.float32(lv)
                ys.append(response_prob_for_inputs(x_row, scaler, rules, coef_mat))
            plt.plot(xs, ys, label=f"{B}={lv:.3g}")

        plt.xlabel(A)
        plt.ylabel("Predicted probability" if USE_SIGMOID else "Predicted score")
        plt.title(f"2D Slice Curves: vary {A}, slices by {B} (others=median)")
        plt.legend()
        plt.tight_layout()
        slice_path = os.path.join(out_dir_slices, f"2d_slices_{A}_by_{B}.png")
        plt.savefig(slice_path, dpi=300)
        plt.close()

        # --- 3D surface for the same pair (A,B) ---
        # Save to main 3D folder (controlled)
        surf_path = os.path.join(out_dir_surfaces, f"3d_surface_{A}_vs_{B}.png")
        plot_3d_surface_for_pair(A, B, X_train_orig, scaler, rules, coef_mat, surf_path)

        # Also copy/save into slice folder, as requested
        if also_save_3d_in_slice_folder:
            surf_in_slice = os.path.join(out_dir_slices, f"3d_surface_{A}_vs_{B}.png")
            # Reuse the same surface computation output by saving again
            # (Cheaper: just copy file. Works across OS.)
            try:
                import shutil
                shutil.copyfile(surf_path, surf_in_slice)
            except Exception:
                # fallback: re-plot (slower but safe)
                plot_3d_surface_for_pair(A, B, X_train_orig, scaler, rules, coef_mat, surf_in_slice)

def plot_3d_surfaces_limited(X_train_orig: pd.DataFrame,
                             scaler: Optional[StandardScaler],
                             rules: List[RuleAntecedent],
                             coef_mat: np.ndarray,
                             out_dir: str,
                             max_pairs: int = MAX_3D_PAIRS):
    pairs = list(itertools.combinations(range(len(FEATURE_COLS)), 2))[:max_pairs]
    for a_idx, b_idx in pairs:
        A = FEATURE_COLS[a_idx]
        B = FEATURE_COLS[b_idx]
        out_path = os.path.join(out_dir, f"3d_surface_{A}_vs_{B}.png")
        plot_3d_surface_for_pair(A, B, X_train_orig, scaler, rules, coef_mat, out_path)

# ==============================
# 3) STEP-BY-STEP EXPORT
# ==============================
def export_step1to5_examples(X_orig: pd.DataFrame,
                            y: pd.Series,
                            scaler: Optional[StandardScaler],
                            rules: List[RuleAntecedent],
                            coef_mat: np.ndarray,
                            out_dir: str,
                            n: int = 10):
    n = min(n, len(X_orig))
    for idx in range(n):
        x_row_orig = X_orig.iloc[idx].values.astype(np.float32)
        x_row_model = scaler.transform(x_row_orig.reshape(1, -1)).astype(np.float32)[0] if scaler is not None else x_row_orig

        score, extra = infer_scores_light(x_row_model.reshape(1, -1), rules, coef_mat, return_rule_mats=True)
        prob = float(sigmoid(score.astype(np.float64))[0]) if USE_SIGMOID else float(score[0])

        firing = extra["firing"][0]
        nf = extra["norm_firing"][0]
        rule_out = extra["rule_out"][0]

        top = np.argsort(nf)[::-1][:TOP_RULES_TO_SHOW]

        sample = {
            "row_index": int(idx),
            "x_original": {FEATURE_COLS[i]: float(x_row_orig[i]) for i in range(len(FEATURE_COLS))},
            "x_model_space": {FEATURE_COLS[i]: float(x_row_model[i]) for i in range(len(FEATURE_COLS))},
            "y_true": int(y.iloc[idx]),
            "score": float(score[0]),
            "probability": prob,
            "step4_top_rules": []
        }

        for rid in top:
            sample["step4_top_rules"].append({
                "rule_id": int(rid),
                "mu": [float(v) for v in rules[rid].mu.tolist()],
                "sigma": [float(v) for v in rules[rid].sigma.tolist()],
                "firing": float(firing[rid]),
                "norm_firing": float(nf[rid]),
                "rule_output": float(rule_out[rid]),
                "weighted_contribution": float(nf[rid] * rule_out[rid]),
            })

        sample["step5_final_output"] = float((nf * rule_out).sum())

        with open(os.path.join(out_dir, f"sample_step1to5_{idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)

# ==============================
# 4) MAIN
# ==============================
def main():
    dirs = make_run_dirs(OUT_ROOT, RUN_NAME)

    tracemalloc.start()
    mem_log = []
    def log_point(tag: str):
        cur, peak = tracemalloc.get_traced_memory()
        mem_log.append({
            "time": now_ts(),
            "tag": tag,
            "tracemalloc_current_gb": bytes_to_gb(cur),
            "tracemalloc_peak_gb": bytes_to_gb(peak),
            "rss_gb": rss_gb()
        })

    log_point("start")

    # Load / split
    X_train_orig, y_train, X_val_orig, y_val, X_test_orig, y_test = load_or_split_data()
    log_point("after_load_data")

    # Scaling
    scaler = None
    if USE_ZSCORE:
        scaler = StandardScaler()
        scaler.fit(X_train_orig.values.astype(np.float32))
        X_train_model = scaler.transform(X_train_orig.values.astype(np.float32)).astype(np.float32)
        X_val_model   = scaler.transform(X_val_orig.values.astype(np.float32)).astype(np.float32)
        X_test_model  = scaler.transform(X_test_orig.values.astype(np.float32)).astype(np.float32)
    else:
        X_train_model = X_train_orig.values.astype(np.float32)
        X_val_model   = X_val_orig.values.astype(np.float32)
        X_test_model  = X_test_orig.values.astype(np.float32)

    log_point("after_scaling")

    # Grid-partition feasibility report (do NOT execute grid rules)
    grid_rules_est = estimate_grid_rules(FEATURE_COLS, MF_COUNTS)
    dtype_bytes = np.dtype(DTYPE).itemsize

    # Provide a conservative estimate: during classic grid inference you often need multiple NxR matrices
    # (firing, norm_firing, rule_out, plus temporaries). We'll estimate 3 big matrices.
    # If you use float64, double these numbers.
    est = {
        "grid_partition_estimated_rules": int(grid_rules_est),
        "assumed_dtype": str(DTYPE),
        "dtype_bytes": int(dtype_bytes),
        "example_batch_sizes": [64, 128, 256, 512, 1024],
        "estimated_memory_gb_for_batch_times_rules_single_matrix": {},
        "estimated_memory_gb_for_batch_times_rules_three_matrices": {},
        "note": "This is an estimation for classic grid-partition inference/aggregation. Actual peak can be higher due to temporaries and Python overhead."
    }
    for b in est["example_batch_sizes"]:
        one = estimate_tensor_gb(b, grid_rules_est, dtype_bytes)
        est["estimated_memory_gb_for_batch_times_rules_single_matrix"][str(b)] = one
        est["estimated_memory_gb_for_batch_times_rules_three_matrices"][str(b)] = one * 3.0

    if USE_GRID_PARTITION:
        raise RuntimeError(
            f"USE_GRID_PARTITION=True would create ~{grid_rules_est:,} rules (grid-partition). "
            "This workflow is intentionally blocked for 11 features."
        )

    # Build K rules
    rules = build_rules_kmeans(X_train_model, K=NUM_RULES)
    log_point("after_kmeans_rules")

    # Train consequents
    firing_train = compute_firing_chunked(X_train_model, rules)
    log_point("after_firing_train")
    coef_mat = train_ridge_consequents(X_train_model, y_train.values.astype(np.float32), firing_train)
    log_point("after_train_consequents")

    # Save rule params / coefs
    antecedents_export = {
        "model_type": "Traditional (Monolithic) TSK-FIS",
        "rule_generation": "KMeans in full feature space (rule reduction; NO grid partition)",
        "use_zscore": bool(USE_ZSCORE),
        "features": FEATURE_COLS,
        "num_rules": int(len(rules)),
        "grid_partition_estimated_rules_if_used": int(grid_rules_est),
        "rules": [{"mu": r.mu.tolist(), "sigma": r.sigma.tolist()} for r in rules]
    }
    with open(os.path.join(dirs["exports"], "antecedents_rules.json"), "w", encoding="utf-8") as f:
        json.dump(antecedents_export, f, indent=2)

    np.save(os.path.join(dirs["exports"], "consequents_coef.npy"), coef_mat)
    with open(os.path.join(dirs["exports"], "consequents_coef.json"), "w", encoding="utf-8") as f:
        json.dump(coef_mat.tolist(), f)

    with open(os.path.join(dirs["exports"], "grid_partition_feasibility.json"), "w", encoding="utf-8") as f:
        json.dump(est, f, indent=2)

    # Inference (memory-aware: no NxR rule_out)
    val_score, _ = infer_scores_light(X_val_model, rules, coef_mat, return_rule_mats=False)
    test_score, _ = infer_scores_light(X_test_model, rules, coef_mat, return_rule_mats=False)
    log_point("after_inference_val_test")

    val_prob = sigmoid(val_score.astype(np.float64)) if USE_SIGMOID else val_score
    test_prob = sigmoid(test_score.astype(np.float64)) if USE_SIGMOID else test_score

    # Export predictions
    def export_preds(X_orig, y_true, score, prob, name):
        dfp = X_orig.copy()
        dfp["y_true"] = y_true.values
        dfp["score"] = score
        dfp["prob"] = prob
        dfp["pred"] = (prob >= THRESHOLD).astype(int)
        dfp.to_csv(os.path.join(dirs["exports"], f"{name}_predictions.csv"), index=False)

    export_preds(X_val_orig, y_val, val_score, val_prob, "val")
    export_preds(X_test_orig, y_test, test_score, test_prob, "test")

    # Metrics + plots
    auc_test = float(roc_auc_score(y_test.values, test_prob))
    cm = confusion_matrix(y_test.values, (test_prob >= THRESHOLD).astype(int))

    plot_roc(y_test.values, test_prob, os.path.join(dirs["figures"], "roc_test.png"))
    plot_confusion(cm, os.path.join(dirs["figures"], "confusion_matrix_test.png"))
    log_point("after_metrics_plots")

    # Visualizations (saved to files)
    plot_2d_single_curves(X_train_orig, scaler, rules, coef_mat, dirs["fig_2d"])
    log_point("after_2d_single_curves")

    # 2D slice curves + 3D surfaces for same pairs (and place surfaces into slice folder)
    plot_2d_slice_curves_with_surface(
        X_train_orig, scaler, rules, coef_mat,
        out_dir_slices=dirs["fig_2d_slices"],
        out_dir_surfaces=dirs["fig_3d"],
        max_pairs=MAX_2D_SLICE_PAIRS,
        also_save_3d_in_slice_folder=True
    )
    log_point("after_2d_slices_and_3d_for_same_pairs")

    # Optional extra 3D surfaces (combinations). If you want ONLY the ones from slice pairs, set MAX_3D_PAIRS=0.
    if MAX_3D_PAIRS > 0:
        plot_3d_surfaces_limited(X_train_orig, scaler, rules, coef_mat, dirs["fig_3d"], max_pairs=MAX_3D_PAIRS)
        log_point("after_additional_3d_surfaces")

    # Step-by-step explain JSONs (small N; safe to allocate NxR rule_out)
    export_step1to5_examples(X_val_orig, y_val, scaler, rules, coef_mat, dirs["explain"], n=EXPLAIN_N)
    log_point("after_explain_exports")

    # Summary + memory report
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_report = {
        "model_type": "Traditional (Monolithic) TSK-FIS",
        "rule_generation": "KMeans (K rules) — avoids grid-partition explosion",
        "features": FEATURE_COLS,
        "num_rules_kmeans": int(len(rules)),
        "grid_partition_estimated_rules_if_used": int(grid_rules_est),
        "dtype_large_mats": str(DTYPE),
        "rule_chunk": int(RULE_CHUNK),
        "ridge_alpha": float(RIDGE_ALPHA),
        "threshold": float(THRESHOLD),
        "test_auc": auc_test,
        "test_confusion_matrix": cm.tolist(),
        "tracemalloc_final_current_gb": bytes_to_gb(cur),
        "tracemalloc_peak_gb": bytes_to_gb(peak),
        "rss_gb_if_available": rss_gb(),
        "runtime_memory_log": mem_log,
        "grid_partition_memory_estimation": est,
        "interpretation_note": (
            "If grid-partition is used with high-dimensional MF counts, R becomes very large and "
            "the aggregation stage requires large (batch_size × R) matrices. This is the dimension trap."
        )
    }
    with open(os.path.join(dirs["exports"], "memory_report.json"), "w", encoding="utf-8") as f:
        json.dump(memory_report, f, indent=2)

    summary = {
        "model_type": "Traditional (Monolithic) TSK-FIS",
        "features": FEATURE_COLS,
        "use_zscore": bool(USE_ZSCORE),
        "num_rules": int(len(rules)),
        "grid_partition_estimated_rules_if_used": int(grid_rules_est),
        "ridge_alpha": float(RIDGE_ALPHA),
        "threshold": float(THRESHOLD),
        "probability_mapping": "sigmoid(score)" if USE_SIGMOID else "raw score",
        "test_auc": auc_test,
        "test_confusion_matrix": cm.tolist(),
        "outputs": {
            "surfaces_2d": dirs["fig_2d"],
            "surfaces_2d_slices": dirs["fig_2d_slices"],
            "surfaces_3d": dirs["fig_3d"],
            "explain_json": dirs["explain"],
            "exports": dirs["exports"]
        }
    }
    with open(os.path.join(dirs["exports"], "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("DONE. Outputs saved to:", dirs["base"])
    print("Memory report:", os.path.join(dirs["exports"], "memory_report.json"))


if __name__ == "__main__":
    main()
