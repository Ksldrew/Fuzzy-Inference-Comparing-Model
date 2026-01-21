# -*- coding: utf-8 -*-
"""
Traditional (Monolithic) TSK-FIS (CPU) — 11 Features
===================================================

This script supports TWO rule-base modes:

(A) Grid-partition (classic "traditional" TSK-FIS):
    - Rules = Cartesian product of MF sets across all features.
    - In high dimensions, rules explode exponentially ("curse of dimensionality / rule explosion"),
      which can cause RAM/time failures during firing-strength computation and normalized aggregation.

(B) KMeans rule reduction (practical workaround for high-D):
    - Rules = K clusters in full feature space (K << grid rules).
    - Keeps the "monolithic inference" pipeline but makes it computable on CPU/RAM-limited machines.

For your 11-feature cardio dataset, full grid-partition is typically infeasible on 32 GB RAM,
so the default behaviour is:
- compute & PRINT the estimated grid rules + estimated RAM,
- then FALL BACK to KMeans rule reduction automatically.

Code comments are in English. Report text can be written in Chinese/English.
"""

import os
import json
import itertools
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

# Optional: better RAM reporting (if installed)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# ==============================
# 0) USER SETTINGS (EDIT HERE)
# ==============================

# --- Dataset folder (your request) ---
DATA_DIR = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

# If you want single-file split instead, set TRAIN/VAL/TEST to "" and set FULL_CSV.
FULL_CSV  = ""  # e.g. r"C:\...\full_dataset.csv"

TARGET_COL = "cardio"

# --- 11 features ---
FEATURE_COLS = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo",
    "cholesterol", "gluc",
    "smoke", "alco", "active"
]

# --- Output folder (your request) ---
OUT_ROOT  = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\feature_11\without_grid"
RUN_NAME  = "traditional_tskfis_cpu_11feat_grid_or_kmeans_v4"

# --- Preprocessing ---
USE_ZSCORE = True
RANDOM_STATE = 42

# --- Rule-base mode ---
# Set USE_GRID_PARTITION=True if you want to TRY classic grid-partition.
# For 11 features, this usually explodes. The script will estimate feasibility first.
USE_GRID_PARTITION = True

# If grid rules are too large, fall back to KMeans automatically (recommended).
FALLBACK_TO_KMEANS_IF_GRID_TOO_LARGE = False  # no rule reduction fallback (grid-only run)

# KMeans rule reduction (fallback / alternative)
NUM_RULES_KMEANS = 120  # typical 50~200

# Membership-function (MF) counts per feature (edit to match your MF plots)
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

# Safety guard: hard cap grid rules to avoid accidental crashes
MAX_GRID_RULES_ALLOWED = 5000

# --- Membership / numerical stability ---
SIGMA_FLOOR = 1e-3

# --- Consequent training ---
RIDGE_ALPHA = 1.0

# --- Probability mapping / classification ---
USE_SIGMOID = True
THRESHOLD = 0.5

# --- Visualization controls ---
GRID_2D = 250
GRID_3D = 45
SURFACE_SLICES_Q = (0.25, 0.75)  # you asked 2 slices
MAX_3D_PAIRS = 12                # limit for runtime


# ==============================
# 1) HELPERS
# ==============================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _derive_age_years_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "age_years" not in df.columns and "age_days" in df.columns:
        df = df.copy()
        df["age_years"] = df["age_days"] / 365.25
    return df

def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).reset_index(drop=True)

def estimate_grid_rules(feature_cols: List[str], mf_counts: Dict[str, int]) -> int:
    R = 1
    for c in feature_cols:
        if c not in mf_counts:
            raise ValueError(f"MF_COUNTS missing feature '{c}'")
        R *= int(mf_counts[c])
    return int(R)

def bytes_to_gb(nbytes: float) -> float:
    return float(nbytes) / (1024.0 ** 3)

def print_ram_status(prefix: str = "") -> Dict[str, float]:
    """
    Print system RAM stats (best-effort). Returns dict for logging to JSON.
    """
    info = {}
    if psutil is not None:
        vm = psutil.virtual_memory()
        info = {
            "total_gb": bytes_to_gb(vm.total),
            "available_gb": bytes_to_gb(vm.available),
            "used_gb": bytes_to_gb(vm.used),
            "percent_used": float(vm.percent),
        }
        print(f"{prefix}RAM total={info['total_gb']:.2f} GB | available={info['available_gb']:.2f} GB | used={info['used_gb']:.2f} GB | used%={info['percent_used']:.1f}%")
    else:
        print(f"{prefix}RAM status: psutil not installed (skipping live RAM stats).")
    return info

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
        "fig_3d_pairs": ensure_dir(os.path.join(base, "figures", "surface3d_with_2slices")),
    }

def load_or_split_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Returns X_train, y_train, X_val, y_val, X_test, y_test in ORIGINAL units.
    """
    if TRAIN_CSV and VAL_CSV and TEST_CSV:
        train_df = _basic_clean(_derive_age_years_if_needed(pd.read_csv(TRAIN_CSV)))
        val_df   = _basic_clean(_derive_age_years_if_needed(pd.read_csv(VAL_CSV)))
        test_df  = _basic_clean(_derive_age_years_if_needed(pd.read_csv(TEST_CSV)))
    else:
        if not FULL_CSV:
            raise ValueError("Provide TRAIN/VAL/TEST CSVs or set FULL_CSV for single-file split.")
        df = _basic_clean(_derive_age_years_if_needed(pd.read_csv(FULL_CSV)))
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

    # Column checks
    for c in FEATURE_COLS + [TARGET_COL]:
        if c not in train_df.columns:
            raise ValueError(f"Missing column '{c}' in TRAIN_CSV")
        if c not in val_df.columns:
            raise ValueError(f"Missing column '{c}' in VAL_CSV")
        if c not in test_df.columns:
            raise ValueError(f"Missing column '{c}' in TEST_CSV")

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].astype(int).copy()
    X_val   = val_df[FEATURE_COLS].copy()
    y_val   = val_df[TARGET_COL].astype(int).copy()
    X_test  = test_df[FEATURE_COLS].copy()
    y_test  = test_df[TARGET_COL].astype(int).copy()
    return X_train, y_train, X_val, y_val, X_test, y_test


@dataclass
class RuleAntecedent:
    mu: np.ndarray      # (F,)
    sigma: np.ndarray   # (F,)


# ==============================
# 2) RULE GENERATION
# ==============================
def build_rules_kmeans(X_train_model: np.ndarray, K: int) -> List[RuleAntecedent]:
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_train_model)
    centers = km.cluster_centers_  # (K,F)

    rules: List[RuleAntecedent] = []
    for k in range(K):
        pts = X_train_model[labels == k]
        sigma = np.std(pts, axis=0) if pts.shape[0] > 1 else np.std(X_train_model, axis=0)
        sigma = np.maximum(sigma, SIGMA_FLOOR)
        rules.append(RuleAntecedent(mu=centers[k].astype(float), sigma=sigma.astype(float)))
    return rules

def _mf_centers_1d(series: pd.Series, m: int) -> np.ndarray:
    """
    Create m MF centers from data distribution (original units).
    For discrete/binary features, this still works (centers will repeat / collapse),
    so we ensure unique-ish spacing by falling back to min/max where needed.
    """
    if m <= 1:
        return np.array([float(series.median())], dtype=float)
    qs = np.linspace(0.05, 0.95, m)
    centers = np.array([float(series.quantile(q)) for q in qs], dtype=float)
    # If collapsed (e.g., binary), spread using unique values
    u = np.unique(series.values.astype(float))
    if len(np.unique(centers)) < min(m, 3) and len(u) <= 5:
        # Use sorted unique values and interpolate to length m
        u = np.sort(u)
        if len(u) == 1:
            centers = np.repeat(u[0], m).astype(float)
        else:
            centers = np.interp(np.linspace(0, len(u)-1, m), np.arange(len(u)), u).astype(float)
    return centers

def build_rules_grid_partition(X_train_orig: pd.DataFrame,
                               scaler: Optional[StandardScaler],
                               feature_cols: List[str],
                               mf_counts: Dict[str, int]) -> List[RuleAntecedent]:
    """
    Classic grid-partition: rules = cartesian product of per-feature MF centers.
    We create Gaussian antecedents:
      mu = centers (converted to model space if scaler exists),
      sigma per feature = median spacing between adjacent centers (or global std fallback).
    """
    centers_list = []
    sigma_list = []

    for feat in feature_cols:
        m = int(mf_counts[feat])
        c = _mf_centers_1d(X_train_orig[feat], m)  # original units
        # sigma: based on center spacing
        if len(c) > 1:
            diffs = np.diff(np.sort(c))
            s = float(np.median(diffs)) if np.any(diffs > 0) else float(np.std(X_train_orig[feat].values))
        else:
            s = float(np.std(X_train_orig[feat].values))
        s = max(s, 1e-6)

        centers_list.append(c)
        sigma_list.append(np.full_like(c, s, dtype=float))

    # Cartesian product of centers -> rule mus in original units
    rule_mus_orig = list(itertools.product(*centers_list))  # length R, each is (F,)
    rules: List[RuleAntecedent] = []

    # Precompute per-feature sigma (use single sigma per feature)
    sigma_per_feat = np.array([float(np.median(s)) if len(s) else 1.0 for s in sigma_list], dtype=float)
    sigma_per_feat = np.maximum(sigma_per_feat, SIGMA_FLOOR)

    for mu_tuple in rule_mus_orig:
        mu_orig = np.array(mu_tuple, dtype=float).reshape(1, -1)
        mu_model = scaler.transform(mu_orig)[0] if scaler is not None else mu_orig[0]
        # sigma should be in same space as mu (model space if scaled)
        if scaler is not None:
            # StandardScaler scales by std; sigma in model space should be sigma/std
            sigma_model = sigma_per_feat / np.maximum(scaler.scale_, 1e-12)
        else:
            sigma_model = sigma_per_feat
        sigma_model = np.maximum(sigma_model, SIGMA_FLOOR)
        rules.append(RuleAntecedent(mu=mu_model.astype(float), sigma=sigma_model.astype(float)))

    return rules


# ==============================
# 3) INFERENCE + TRAINING
# ==============================
def compute_firing(X_model: np.ndarray, rules: List[RuleAntecedent], chunk: int = 40) -> np.ndarray:
    """
    firing[n,r] = Π_j exp(-0.5*((x_j - mu_rj)/sigma_rj)^2)
    computed in chunks to limit RAM.
    """
    N, F = X_model.shape
    R = len(rules)
    firing = np.zeros((N, R), dtype=float)

    for start in range(0, R, chunk):
        end = min(start + chunk, R)
        mus = np.stack([rules[r].mu for r in range(start, end)], axis=0)        # (Rc,F)
        sig = np.stack([rules[r].sigma for r in range(start, end)], axis=0)     # (Rc,F)

        x = X_model[:, None, :]                 # (N,1,F)
        m = mus[None, :, :]                     # (1,Rc,F)
        s = np.maximum(sig[None, :, :], SIGMA_FLOOR)

        mf = np.exp(-0.5 * ((x - m) / s) ** 2)  # (N,Rc,F)
        firing[:, start:end] = np.prod(mf, axis=2)

    return firing

def normalize_firing(firing: np.ndarray) -> np.ndarray:
    return firing / (firing.sum(axis=1, keepdims=True) + 1e-12)

def train_ridge_consequents(X_model: np.ndarray, y: np.ndarray, firing: np.ndarray) -> np.ndarray:
    N, F = X_model.shape
    R = firing.shape[1]
    X_ext = np.hstack([X_model, np.ones((N, 1), dtype=float)])
    coef_mat = np.zeros((R, F + 1), dtype=float)

    for r in range(R):
        w = firing[:, r]
        if np.all(w < 1e-12):
            continue
        model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
        model.fit(X_ext, y, sample_weight=w)
        coef_mat[r, :] = model.coef_
        coef_mat[r, -1] += model.intercept_
    return coef_mat

def infer_scores(X_model: np.ndarray, rules: List[RuleAntecedent], coef_mat: np.ndarray) -> np.ndarray:
    firing = compute_firing(X_model, rules)
    nf = normalize_firing(firing)
    X_ext = np.hstack([X_model, np.ones((X_model.shape[0], 1), dtype=float)])
    rule_out = X_ext @ coef_mat.T
    return (nf * rule_out).sum(axis=1)


# ==============================
# 4) VISUALIZATION: 3D surface + 2 slice curves in SAME folder per pair
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

def response_prob(x_orig_row: np.ndarray,
                  scaler: Optional[StandardScaler],
                  rules: List[RuleAntecedent],
                  coef_mat: np.ndarray) -> float:
    x = x_orig_row.reshape(1, -1).astype(float)
    x_model = scaler.transform(x) if scaler is not None else x
    score = infer_scores(x_model, rules, coef_mat)
    if USE_SIGMOID:
        return float(sigmoid(score)[0])
    return float(score[0])

def plot_surface_with_two_slices(X_train_orig: pd.DataFrame,
                                 scaler: Optional[StandardScaler],
                                 rules: List[RuleAntecedent],
                                 coef_mat: np.ndarray,
                                 out_pair_dir: str,
                                 a_idx: int,
                                 b_idx: int):
    """
    Save 3 images in out_pair_dir:
      1) surface_3d.png
      2) slice_q25.png   (vary A, fix B at 25% quantile)
      3) slice_q75.png   (vary A, fix B at 75% quantile)
    """
    ensure_dir(out_pair_dir)
    A = FEATURE_COLS[a_idx]
    B = FEATURE_COLS[b_idx]

    med = X_train_orig[FEATURE_COLS].median().to_dict()
    xA = linspace_in_range(X_train_orig[A], GRID_3D)
    xB = linspace_in_range(X_train_orig[B], GRID_3D)
    XA, XB = np.meshgrid(xA, xB)

    Z = np.zeros_like(XA, dtype=float)
    for i in range(XA.shape[0]):
        for j in range(XA.shape[1]):
            row = np.array([med[c] for c in FEATURE_COLS], dtype=float)
            row[a_idx] = XA[i, j]
            row[b_idx] = XB[i, j]
            Z[i, j] = response_prob(row, scaler, rules, coef_mat)

    # (1) 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XA, XB, Z)
    ax.set_xlabel(A)
    ax.set_ylabel(B)
    ax.set_zlabel("Prob" if USE_SIGMOID else "Score")
    ax.set_title(f"3D Surface: {A} vs {B} (others=median)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_pair_dir, "surface_3d.png"), dpi=300)
    plt.close(fig)

    # (2) two slice curves
    xs = linspace_in_range(X_train_orig[A], GRID_2D)
    q_low, q_high = SURFACE_SLICES_Q
    b_low = float(X_train_orig[B].quantile(q_low))
    b_high = float(X_train_orig[B].quantile(q_high))

    for name, b_val in [(f"slice_q{int(q_low*100)}.png", b_low), (f"slice_q{int(q_high*100)}.png", b_high)]:
        ys = []
        for v in xs:
            row = np.array([med[c] for c in FEATURE_COLS], dtype=float)
            row[a_idx] = v
            row[b_idx] = b_val
            ys.append(response_prob(row, scaler, rules, coef_mat))
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(A)
        plt.ylabel("Predicted probability" if USE_SIGMOID else "Predicted score")
        plt.title(f"2D Slice: vary {A}, fix {B}={b_val:.3g} (others=median)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_pair_dir, name), dpi=300)
        plt.close()

def generate_limited_surface_folders(X_train_orig: pd.DataFrame,
                                     scaler: Optional[StandardScaler],
                                     rules: List[RuleAntecedent],
                                     coef_mat: np.ndarray,
                                     out_root_dir: str,
                                     max_pairs: int = MAX_3D_PAIRS):
    pairs = list(itertools.combinations(range(len(FEATURE_COLS)), 2))[:max_pairs]
    for a_idx, b_idx in pairs:
        A = FEATURE_COLS[a_idx]
        B = FEATURE_COLS[b_idx]
        pair_dir = os.path.join(out_root_dir, f"{A}_vs_{B}")
        plot_surface_with_two_slices(X_train_orig, scaler, rules, coef_mat, pair_dir, a_idx, b_idx)


# ==============================
# 5) MAIN
# ==============================
def main():
    dirs = make_run_dirs(OUT_ROOT, RUN_NAME)

    print("\n=== Traditional TSK-FIS (CPU) | 11 Features ===")
    print("Output folder:", dirs["base"])
    live_ram = print_ram_status(prefix="[BEFORE] ")

    # Load data
    X_train_orig, y_train, X_val_orig, y_val, X_test_orig, y_test = load_or_split_data()

    # Scaling
    scaler = None
    if USE_ZSCORE:
        scaler = StandardScaler()
        scaler.fit(X_train_orig.values.astype(float))
        X_train_model = scaler.transform(X_train_orig.values.astype(float))
        X_val_model   = scaler.transform(X_val_orig.values.astype(float))
        X_test_model  = scaler.transform(X_test_orig.values.astype(float))
    else:
        X_train_model = X_train_orig.values.astype(float)
        X_val_model   = X_val_orig.values.astype(float)
        X_test_model  = X_test_orig.values.astype(float)

    # --- Grid feasibility report ---
    grid_rules_est = estimate_grid_rules(FEATURE_COLS, MF_COUNTS)
    # Memory estimate: firing tensor roughly N * R * 8 bytes (float64) + overhead
    N_test = int(X_test_model.shape[0])
    est_firing_bytes = float(N_test) * float(grid_rules_est) * 8.0
    est_firing_gb = bytes_to_gb(est_firing_bytes)

    feasibility = {
        "use_grid_partition_requested": bool(USE_GRID_PARTITION),
        "estimated_grid_rules": int(grid_rules_est),
        "estimated_firing_matrix_gb_test_only": float(est_firing_gb),
        "max_grid_rules_allowed": int(MAX_GRID_RULES_ALLOWED),
        "note": "Grid-partition rule explosion increases RAM/time; for 11 features it is usually infeasible on CPU."
    }
    feasibility["live_ram_before"] = live_ram
    with open(os.path.join(dirs["exports"], "grid_partition_feasibility.json"), "w", encoding="utf-8") as f:
        json.dump(feasibility, f, indent=2)

    print(f"\n[GRID CHECK] Estimated grid rules R = {grid_rules_est:,}")
    print(f"[GRID CHECK] Estimated firing matrix (test only) ~ {est_firing_gb:.2f} GB (float64, N={N_test})")
    print(f"[GRID CHECK] Safety cap MAX_GRID_RULES_ALLOWED = {MAX_GRID_RULES_ALLOWED:,}")

    # Decide rule mode
    rule_mode = "kmeans"
    if USE_GRID_PARTITION and grid_rules_est <= MAX_GRID_RULES_ALLOWED:
        rule_mode = "grid"
        print("[GRID CHECK] Grid rules within cap -> building GRID-PARTITION rule base...")
    else:
        if USE_GRID_PARTITION and grid_rules_est > MAX_GRID_RULES_ALLOWED:
            print("[GRID CHECK] Grid rules exceed cap -> grid-partition is blocked to prevent RAM/time failure.")
            if FALLBACK_TO_KMEANS_IF_GRID_TOO_LARGE:
                print("[GRID CHECK] Falling back to KMeans rule reduction.")
            else:
                raise RuntimeError("Grid-partition blocked (rule explosion). Set FALLBACK_TO_KMEANS_IF_GRID_TOO_LARGE = False  # no rule reduction fallback (grid-only run) to proceed.")

    # Build rules (grid-only)
    if rule_mode != "grid":
        raise RuntimeError("Grid-partition was not selected. This grid-only script disables rule reduction. Reduce MF counts / features or raise MAX_GRID_RULES_ALLOWED at your own risk.")
    rules = build_rules_grid_partition(X_train_orig, scaler, FEATURE_COLS, MF_COUNTS)

    print(f"\n[RULES] Mode = {rule_mode.upper()} | Rules = {len(rules):,}")

    # Train consequents
    print("[TRAIN] Computing firing strengths on train set (chunked)...")
    firing_train = compute_firing(X_train_model, rules, chunk=40)
    print("[TRAIN] Training Ridge consequents...")
    coef_mat = train_ridge_consequents(X_train_model, y_train.values.astype(float), firing_train)

    np.save(os.path.join(dirs["exports"], "consequents_coef.npy"), coef_mat)
    with open(os.path.join(dirs["exports"], "consequents_coef.json"), "w", encoding="utf-8") as f:
        json.dump(coef_mat.tolist(), f)

    # Inference
    print("[EVAL] Inference on validation/test...")
    val_score = infer_scores(X_val_model, rules, coef_mat)
    test_score = infer_scores(X_test_model, rules, coef_mat)

    val_prob = sigmoid(val_score) if USE_SIGMOID else val_score
    test_prob = sigmoid(test_score) if USE_SIGMOID else test_score

    # Metrics
    auc_test = float(roc_auc_score(y_test.values, test_prob))
    cm = confusion_matrix(y_test.values, (test_prob >= THRESHOLD).astype(int))

    print(f"[METRICS] Test AUC = {auc_test:.4f}")
    print(f"[METRICS] Confusion Matrix =\n{cm}")

    # Save plots
    plot_roc(y_test.values, test_prob, os.path.join(dirs["figures"], "roc_test.png"))
    plot_confusion(cm, os.path.join(dirs["figures"], "confusion_matrix_test.png"))

    # Save per-pair folders: 1x3D + 2x2D slices (your requirement)
    print("[VIZ] Generating per-pair surface folders (3 images each)...")
    generate_limited_surface_folders(X_train_orig, scaler, rules, coef_mat, dirs["fig_3d_pairs"], max_pairs=MAX_3D_PAIRS)

    # Summary export
    live_ram_after = print_ram_status(prefix="[AFTER] ")
    summary = {
        "model_type": "Traditional (Monolithic) TSK-FIS",
        "features": FEATURE_COLS,
        "rule_mode_used": rule_mode,
        "num_rules_used": int(len(rules)),
        "use_zscore": bool(USE_ZSCORE),
        "ridge_alpha": float(RIDGE_ALPHA),
        "threshold": float(THRESHOLD),
        "probability_mapping": "sigmoid(score)" if USE_SIGMOID else "raw score",
        "test_auc": auc_test,
        "test_confusion_matrix": cm.tolist(),
        "grid_rules_estimated": int(grid_rules_est),
        "grid_estimated_firing_matrix_gb_test_only": float(est_firing_gb),
        "live_ram_after": live_ram_after,
        "interpretation_note": "AUC reflects ranking ability; threshold controls sensitivity vs specificity. Grid-partition in 11D usually causes rule explosion, so KMeans rule reduction is used for feasibility."
    }
    with open(os.path.join(dirs["exports"], "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE. Outputs saved to:", dirs["base"])


if __name__ == "__main__":
    main()
