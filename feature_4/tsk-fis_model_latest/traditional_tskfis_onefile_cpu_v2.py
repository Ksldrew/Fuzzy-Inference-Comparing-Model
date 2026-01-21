
# -*- coding: utf-8 -*-
"""
Traditional (Monolithic) TSK-FIS (CPU) — KMeans + 3 Gaussian MFs + full rule grid + one-shot Ridge consequents.

This script is designed to match your Chapter 3 "traditional TSK-FIS" description:
- MF generation: unsupervised KMeans per feature (3 clusters -> Low/Normal/High)
- Rule base: fixed grid partition (3^4 = 81 rules for 4 features)
- Consequents: trained once via (weighted) Ridge regression (no epochs, no BCE loss)
- Inference: normalized firing strengths + weighted sum of rule outputs

Extra (for report-ready documentation):
- 2D response curves (single-variable + 3-slice curves per pair)
- 3D response surfaces per feature pair
- MF plots in original units (always)
- Optional MF plots in z-score space (only if USE_ZSCORE=True)
- Histogram + MF overlay (original units)

Author: generated for FYP documentation
"""

import os
import json
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ==============================
# 0) USER SETTINGS (EDIT HERE)
# ==============================
TRAIN_CSV = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\train.csv"
VAL_CSV   = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\val.csv"
TEST_CSV  = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\test.csv"

OUT_ROOT  = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\feature_4\tsk-fis_model"
RUN_NAME  = "traditional_tskfis_cpu"

TARGET_COL = "cardio"

# Your chosen 4 features (fixed):
FEATURE_COLS = ["ap_hi", "ap_lo", "age_years", "cholesterol"]

# Traditional baseline: set this False if you want "no z-score" in your report
# (KMeans here is per-feature (1D), so z-score is optional; see notes in your report.)
USE_ZSCORE = False

# MF / rule settings
MF_PER_FEATURE = 3  # fixed 3 -> Low/Normal/High (matches Chapter 3)
KMEANS_RANDOM_STATE = 42
SIGMA_FLOOR = 1e-3  # prevents divide-by-zero

# Inference/probability
USE_SIGMOID = True  # convert score to [0,1] probability for ROC/CM
THRESHOLD = 0.5     # classification threshold for confusion matrix

# Visualization density
GRID_2D = 250               # points for 2D curves
GRID_3D = 45                # resolution for 3D surface grid (increase carefully)
SLICE_LEVELS = [0.25, 0.50, 0.75]  # for 2D slice curves: low/med/high (quantiles)

# How many samples to export step-by-step explain (JSON)
EXPLAIN_N = 10


# ==============================
# 1) HELPERS
# ==============================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def gaussian_mf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), SIGMA_FLOOR)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

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
        "fig_mf": ensure_dir(os.path.join(base, "figures", "mf")),
        "fig_overlay": ensure_dir(os.path.join(base, "figures", "mf_overlay")),
        "fig_2d": ensure_dir(os.path.join(base, "figures", "surfaces_2d")),
        "fig_2d_slices": ensure_dir(os.path.join(base, "figures", "surfaces_2d_slices")),
        "fig_3d": ensure_dir(os.path.join(base, "figures", "surfaces_3d")),
        "explain": ensure_dir(os.path.join(base, "step1to5_explain")),
    }

def load_split(path: str, features: List[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    X = df[features].copy()
    y = df[target].astype(int).copy()
    return X, y

@dataclass
class MFParams:
    mu: float
    sigma: float

def kmeans_mf_params_1d(x_col: np.ndarray, n_mf: int = 3) -> List[MFParams]:
    """
    KMeans clustering on a single feature column to obtain MF centers (mu).
    Sigma is estimated from within-cluster std.
    """
    x = x_col.reshape(-1, 1).astype(float)
    km = KMeans(n_clusters=n_mf, random_state=KMEANS_RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(x)
    centers = km.cluster_centers_.reshape(-1)

    # sort by center => Low/Normal/High order
    order = np.argsort(centers)
    centers = centers[order]

    params: List[MFParams] = []
    for ci, c in enumerate(centers):
        # find points in this cluster (after re-order)
        original_cluster = order[ci]
        pts = x_col[labels == original_cluster].astype(float)
        if len(pts) <= 1:
            sigma = np.std(x_col.astype(float)) if np.std(x_col.astype(float)) > 0 else 1.0
        else:
            sigma = float(np.std(pts))
        sigma = max(sigma, SIGMA_FLOOR)
        params.append(MFParams(mu=float(c), sigma=float(sigma)))
    return params

def build_rule_grid(n_features: int, mfs_per_feature: int) -> np.ndarray:
    """
    Full grid partition: all MF index combinations.
    Example: 4 features, 3 MFs => 3^4 = 81 rules.
    """
    grid = list(itertools.product(range(mfs_per_feature), repeat=n_features))
    return np.array(grid, dtype=int)

def compute_memberships(X_model: np.ndarray, mf_dict: Dict[str, List[MFParams]], feature_cols: List[str]) -> np.ndarray:
    """
    Return memberships: shape (N, F, M)
    """
    N = X_model.shape[0]
    F = len(feature_cols)
    M = MF_PER_FEATURE
    mus = np.zeros((N, F, M), dtype=float)
    for j, feat in enumerate(feature_cols):
        x = X_model[:, j]
        for m, p in enumerate(mf_dict[feat]):
            mus[:, j, m] = gaussian_mf(x, p.mu, p.sigma)
    return mus

def compute_firing(mus: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """
    mus: (N,F,M)
    rules: (R,F) each entry is MF index
    output: firing (N,R)
    """
    N, F, M = mus.shape
    R = rules.shape[0]
    firing = np.ones((N, R), dtype=float)
    for j in range(F):
        firing *= mus[:, j, rules[:, j]]
    return firing

def normalize_firing(firing: np.ndarray) -> np.ndarray:
    denom = firing.sum(axis=1, keepdims=True) + 1e-12
    return firing / denom

def train_ridge_consequents(X_model: np.ndarray, y: np.ndarray, firing: np.ndarray) -> np.ndarray:
    """
    Train one linear consequent per rule using weighted Ridge regression.
    consequent: [p,q,r,s,t] for 4 features + bias
    returns coef_mat shape (R, F+1)
    """
    N, F = X_model.shape
    R = firing.shape[1]
    X_ext = np.hstack([X_model, np.ones((N, 1), dtype=float)])  # add bias
    coef_mat = np.zeros((R, F + 1), dtype=float)

    for r in range(R):
        w = firing[:, r]
        if np.all(w < 1e-12):
            # never fires => keep zeros
            continue
        model = Ridge(alpha=1.0, random_state=KMEANS_RANDOM_STATE)
        model.fit(X_ext, y, sample_weight=w)
        coef_mat[r, :] = model.coef_
        coef_mat[r, -1] += model.intercept_  # sklearn stores intercept separately
    return coef_mat

def infer_scores(X_model: np.ndarray, mf_dict: Dict[str, List[MFParams]], rules: np.ndarray, coef_mat: np.ndarray, feature_cols: List[str]):
    mus = compute_memberships(X_model, mf_dict, feature_cols)
    firing = compute_firing(mus, rules)
    nf = normalize_firing(firing)
    X_ext = np.hstack([X_model, np.ones((X_model.shape[0], 1), dtype=float)])
    rule_out = X_ext @ coef_mat.T  # (N,R)
    score = (nf * rule_out).sum(axis=1)
    extra = {"mus": mus, "firing": firing, "norm_firing": nf, "rule_out": rule_out}
    return score, extra


# ==============================
# 2) VISUALIZATION
# ==============================
def plot_mfs_original(feature: str, mf_params_orig: List[MFParams], series_orig: pd.Series, out_path: str):
    xs = linspace_in_range(series_orig, 400)
    plt.figure()
    for p in mf_params_orig:
        plt.plot(xs, gaussian_mf(xs, p.mu, p.sigma))
    plt.xlabel(feature)
    plt.ylabel("Membership degree")
    plt.title(f"Gaussian MFs in original units: {feature}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_mfs_zscore(feature: str, mf_params_z: List[MFParams], out_path: str):
    xs = np.linspace(-4, 4, 400)
    plt.figure()
    for p in mf_params_z:
        plt.plot(xs, gaussian_mf(xs, p.mu, p.sigma))
    plt.xlabel(f"{feature} (z-score)")
    plt.ylabel("Membership degree")
    plt.title(f"Gaussian MFs in z-score space: {feature}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_hist_overlay(feature: str, series_orig: pd.Series, mf_params_orig: List[MFParams], out_path: str):
    xs = linspace_in_range(series_orig, 400)
    plt.figure()
    plt.hist(series_orig.dropna().values, bins=40, density=True, alpha=0.4)
    for p in mf_params_orig:
        plt.plot(xs, gaussian_mf(xs, p.mu, p.sigma))
    plt.xlabel(feature)
    plt.ylabel("Density / Membership")
    plt.title(f"Histogram + MF overlay: {feature}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_roc(y_true: np.ndarray, prob: np.ndarray, out_path: str):
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={auc:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_confusion(cm: np.ndarray, out_path: str):
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["True 0","True 1"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def response_prob_for_inputs(x_orig_row: np.ndarray,
                             feature_cols: List[str],
                             scaler: StandardScaler,
                             mf_dict_model: Dict[str, List[MFParams]],
                             rules: np.ndarray,
                             coef_mat: np.ndarray) -> float:
    """
    x_orig_row: (F,) in original units
    """
    x = x_orig_row.reshape(1, -1).astype(float)
    if scaler is not None:
        x_model = scaler.transform(x)
    else:
        x_model = x
    score, _ = infer_scores(x_model, mf_dict_model, rules, coef_mat, feature_cols)
    if USE_SIGMOID:
        return float(sigmoid(score)[0])
    return float(score[0])

def plot_2d_single_curves(X_train_orig: pd.DataFrame,
                          feature_cols: List[str],
                          scaler: StandardScaler,
                          mf_dict_model: Dict[str, List[MFParams]],
                          rules: np.ndarray,
                          coef_mat: np.ndarray,
                          out_dir: str):
    """
    For each feature xi: vary xi across range, fix other features at train median.
    Plot x vs y (probability).
    """
    med = X_train_orig[feature_cols].median().values.astype(float)

    for i, feat in enumerate(feature_cols):
        xs = linspace_in_range(X_train_orig[feat], GRID_2D)
        ys = []
        for v in xs:
            x_row = med.copy()
            x_row[i] = v
            ys.append(response_prob_for_inputs(x_row, feature_cols, scaler, mf_dict_model, rules, coef_mat))
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(feat)
        plt.ylabel("Predicted probability" if USE_SIGMOID else "Predicted score")
        plt.title(f"2D Response Curve: {feat} (others fixed at median)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"2d_curve_{feat}.png"), dpi=300)
        plt.close()

def plot_2d_slice_curves_per_pair(X_train_orig: pd.DataFrame,
                                  feature_cols: List[str],
                                  scaler: StandardScaler,
                                  mf_dict_model: Dict[str, List[MFParams]],
                                  rules: np.ndarray,
                                  coef_mat: np.ndarray,
                                  out_dir: str):
    """
    For each ordered pair (x-axis feature A, slicing feature B):
    - vary A on x-axis
    - fix B at 3 quantiles (25/50/75), producing 3 lines
    - fix remaining features at train median
    This gives a "2D plot with 3 parts" analogous to a 3D surface slice.
    """
    med = X_train_orig[feature_cols].median().to_dict()

    for a_idx, b_idx in itertools.permutations(range(len(feature_cols)), 2):
        A = feature_cols[a_idx]
        B = feature_cols[b_idx]

        xs = linspace_in_range(X_train_orig[A], GRID_2D)
        b_levels = [float(X_train_orig[B].quantile(q)) for q in SLICE_LEVELS]

        plt.figure()
        for lv in b_levels:
            ys = []
            for v in xs:
                x_row = np.array([med[c] for c in feature_cols], dtype=float)
                x_row[a_idx] = v
                x_row[b_idx] = lv
                ys.append(response_prob_for_inputs(x_row, feature_cols, scaler, mf_dict_model, rules, coef_mat))
            plt.plot(xs, ys, label=f"{B}={lv:.3g}")

        plt.xlabel(A)
        plt.ylabel("Predicted probability" if USE_SIGMOID else "Predicted score")
        plt.title(f"2D Slice Curves: vary {A}, slices by {B} (others=median)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"2d_slices_{A}_by_{B}.png"), dpi=300)
        plt.close()

def plot_3d_surfaces_per_pair(X_train_orig: pd.DataFrame,
                              feature_cols: List[str],
                              scaler: StandardScaler,
                              mf_dict_model: Dict[str, List[MFParams]],
                              rules: np.ndarray,
                              coef_mat: np.ndarray,
                              out_dir: str):
    """
    For each feature pair (A,B):
    - vary A and B on a grid
    - fix others at train median
    - plot surface z = predicted probability/score
    """
    med = X_train_orig[feature_cols].median().to_dict()

    pairs = list(itertools.combinations(range(len(feature_cols)), 2))
    for a_idx, b_idx in pairs:
        A = feature_cols[a_idx]
        B = feature_cols[b_idx]

        xA = linspace_in_range(X_train_orig[A], GRID_3D)
        xB = linspace_in_range(X_train_orig[B], GRID_3D)
        XA, XB = np.meshgrid(xA, xB)

        Z = np.zeros_like(XA, dtype=float)
        for i in range(XA.shape[0]):
            for j in range(XA.shape[1]):
                x_row = np.array([med[c] for c in feature_cols], dtype=float)
                x_row[a_idx] = XA[i, j]
                x_row[b_idx] = XB[i, j]
                Z[i, j] = response_prob_for_inputs(x_row, feature_cols, scaler, mf_dict_model, rules, coef_mat)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(XA, XB, Z)
        ax.set_xlabel(A)
        ax.set_ylabel(B)
        ax.set_zlabel("Prob" if USE_SIGMOID else "Score")
        ax.set_title(f"3D Surface: {A} vs {B} (others=median)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"3d_surface_{A}_vs_{B}.png"), dpi=300)
        plt.close(fig)


# ==============================
# 3) STEP-BY-STEP EXPORT
# ==============================
def export_step1to5_examples(X_orig: pd.DataFrame,
                            y: pd.Series,
                            scaler: StandardScaler,
                            feature_cols: List[str],
                            mf_dict_model: Dict[str, List[MFParams]],
                            rules: np.ndarray,
                            coef_mat: np.ndarray,
                            out_dir: str,
                            n: int = 10):
    """
    Export JSON explaining the calculations (Step1..Step5) for the first n samples.
    """
    n = min(n, len(X_orig))
    for idx in range(n):
        x_row_orig = X_orig.iloc[idx].values.astype(float)
        x_row_model = scaler.transform(x_row_orig.reshape(1, -1))[0] if scaler is not None else x_row_orig

        score, extra = infer_scores(x_row_model.reshape(1, -1), mf_dict_model, rules, coef_mat, feature_cols)
        prob = float(sigmoid(score)[0]) if USE_SIGMOID else float(score[0])

        sample = {
            "row_index": int(idx),
            "x_original": {feature_cols[i]: float(x_row_orig[i]) for i in range(len(feature_cols))},
            "x_model_space": {feature_cols[i]: float(x_row_model[i]) for i in range(len(feature_cols))},
            "y_true": int(y.iloc[idx]),
            "score": float(score[0]),
            "probability": prob,
            "step3_membership_mu": {
                feature_cols[i]: [float(v) for v in extra["mus"][0, i, :].tolist()]
                for i in range(len(feature_cols))
            },
        }

        # include top-10 fired rules for readability
        firing = extra["firing"][0]
        nf = extra["norm_firing"][0]
        rule_out = extra["rule_out"][0]
        top = np.argsort(nf)[::-1][:10]

        sample["step4_top_rules"] = []
        for rid in top:
            sample["step4_top_rules"].append({
                "rule_id": int(rid),
                "mf_indices": [int(v) for v in rules[rid].tolist()],
                "firing": float(firing[rid]),
                "norm_firing": float(nf[rid]),
                "rule_output": float(rule_out[rid]),
                "weighted_contribution": float(nf[rid] * rule_out[rid])
            })

        # step5 final
        sample["step5_final_output"] = float((nf * rule_out).sum())

        with open(os.path.join(out_dir, f"sample_step1to5_{idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)


# ==============================
# 4) MAIN
# ==============================
def main():
    dirs = make_run_dirs(OUT_ROOT, RUN_NAME)

    # Load data
    X_train_orig, y_train = load_split(TRAIN_CSV, FEATURE_COLS, TARGET_COL)
    X_val_orig, y_val = load_split(VAL_CSV, FEATURE_COLS, TARGET_COL)
    X_test_orig, y_test = load_split(TEST_CSV, FEATURE_COLS, TARGET_COL)

    # Optional z-score preprocessing
    scaler = None
    if USE_ZSCORE:
        scaler = StandardScaler()
        scaler.fit(X_train_orig.values.astype(float))
        X_train_model = scaler.transform(X_train_orig.values.astype(float))
        X_val_model = scaler.transform(X_val_orig.values.astype(float))
        X_test_model = scaler.transform(X_test_orig.values.astype(float))
    else:
        X_train_model = X_train_orig.values.astype(float)
        X_val_model = X_val_orig.values.astype(float)
        X_test_model = X_test_orig.values.astype(float)

    # MF generation via KMeans (per feature)
    mf_dict_model: Dict[str, List[MFParams]] = {}
    mf_dict_orig: Dict[str, List[MFParams]] = {}

    for j, feat in enumerate(FEATURE_COLS):
        # model-space MF (for inference)
        params_model = kmeans_mf_params_1d(X_train_model[:, j], n_mf=MF_PER_FEATURE)
        mf_dict_model[feat] = params_model

        # original-unit MF (for plots)
        if scaler is not None:
            # inverse transform of mu/sigma approximately:
            # x_orig = z * std + mean; sigma scales by std
            mean = float(scaler.mean_[j])
            std = float(np.sqrt(scaler.var_[j]))
            params_orig = [MFParams(mu=p.mu * std + mean, sigma=p.sigma * std) for p in params_model]
        else:
            params_orig = [MFParams(mu=p.mu, sigma=p.sigma) for p in params_model]
        mf_dict_orig[feat] = params_orig

    # Export MF params
    mf_export = {
        "use_zscore": bool(USE_ZSCORE),
        "features": FEATURE_COLS,
        "mf_per_feature": MF_PER_FEATURE,
        "mf_params_model_space": {
            f: [{"mu": p.mu, "sigma": p.sigma} for p in mf_dict_model[f]] for f in FEATURE_COLS
        },
        "mf_params_original_units": {
            f: [{"mu": p.mu, "sigma": p.sigma} for p in mf_dict_orig[f]] for f in FEATURE_COLS
        }
    }
    with open(os.path.join(dirs["exports"], "mf_params.json"), "w", encoding="utf-8") as f:
        json.dump(mf_export, f, indent=2)

    # Plot MFs + overlays
    for feat in FEATURE_COLS:
        plot_mfs_original(feat, mf_dict_orig[feat], X_train_orig[feat], os.path.join(dirs["fig_mf"], f"mf_original_{feat}.png"))
        plot_hist_overlay(feat, X_train_orig[feat], mf_dict_orig[feat], os.path.join(dirs["fig_overlay"], f"mf_overlay_{feat}.png"))
        if USE_ZSCORE:
            plot_mfs_zscore(feat, mf_dict_model[feat], os.path.join(dirs["fig_mf"], f"mf_zscore_{feat}.png"))

    # Build rules (truth-table style grid)
    rules = build_rule_grid(n_features=len(FEATURE_COLS), mfs_per_feature=MF_PER_FEATURE)
    pd.DataFrame(rules, columns=[f"{c}_mf_idx" for c in FEATURE_COLS]).to_csv(
        os.path.join(dirs["exports"], "rule_grid.csv"), index=False
    )

    # Train consequents (one-shot Ridge)
    mus_train = compute_memberships(X_train_model, mf_dict_model, FEATURE_COLS)
    firing_train = compute_firing(mus_train, rules)
    coef_mat = train_ridge_consequents(X_train_model, y_train.values, firing_train)

    np.save(os.path.join(dirs["exports"], "consequents_coef.npy"), coef_mat)
    with open(os.path.join(dirs["exports"], "consequents_coef.json"), "w", encoding="utf-8") as f:
        json.dump(coef_mat.tolist(), f)

    # Inference
    val_score, _ = infer_scores(X_val_model, mf_dict_model, rules, coef_mat, FEATURE_COLS)
    test_score, _ = infer_scores(X_test_model, mf_dict_model, rules, coef_mat, FEATURE_COLS)

    val_prob = sigmoid(val_score) if USE_SIGMOID else val_score
    test_prob = sigmoid(test_score) if USE_SIGMOID else test_score

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

    # Metrics + plots (test as main)
    auc_test = float(roc_auc_score(y_test.values, test_prob))
    cm = confusion_matrix(y_test.values, (test_prob >= THRESHOLD).astype(int))
    plot_roc(y_test.values, test_prob, os.path.join(dirs["figures"], "roc_test.png"))
    plot_confusion(cm, os.path.join(dirs["figures"], "confusion_matrix_test.png"))

    # 2D (single-variable) + 2D slice curves (3 lines) + 3D surfaces
    plot_2d_single_curves(X_train_orig, FEATURE_COLS, scaler, mf_dict_model, rules, coef_mat, dirs["fig_2d"])
    plot_2d_slice_curves_per_pair(X_train_orig, FEATURE_COLS, scaler, mf_dict_model, rules, coef_mat, dirs["fig_2d_slices"])
    plot_3d_surfaces_per_pair(X_train_orig, FEATURE_COLS, scaler, mf_dict_model, rules, coef_mat, dirs["fig_3d"])

    # Step-by-step explain JSONs (first N samples from val)
    export_step1to5_examples(X_val_orig, y_val, scaler, FEATURE_COLS, mf_dict_model, rules, coef_mat, dirs["explain"], n=EXPLAIN_N)

    # Summary
    summary = {
        "model_type": "Traditional (Monolithic) TSK-FIS",
        "use_zscore": bool(USE_ZSCORE),
        "features": FEATURE_COLS,
        "mf_per_feature": MF_PER_FEATURE,
        "num_rules": int(rules.shape[0]),
        "test_auc": auc_test,
        "test_confusion_matrix": cm.tolist(),
        "threshold": float(THRESHOLD),
        "probability_mapping": "sigmoid(score)" if USE_SIGMOID else "raw score",
        "notes": {
            "why_zscore_optional": "KMeans is applied per feature (1D), so global feature scaling is optional. Use z-score if you want numerical stability and comparable parameter ranges; disable it if your report defines z-score only for the modular model."
        }
    }
    with open(os.path.join(dirs["exports"], "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("DONE. Outputs saved to:", dirs["base"])


if __name__ == "__main__":
    main()
