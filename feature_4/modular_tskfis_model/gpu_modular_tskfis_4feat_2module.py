# -*- coding: utf-8 -*-
"""
GPU Modular TSK-FIS (4 features, 2 modules)
- Features: ap_hi, ap_lo, age_years, cholesterol
- Modules: (age_years + cholesterol), (ap_hi + ap_lo)
- Z-score normalization (fit on train only)
- MF init by KMeans in Z-space
- End-to-end training with BCEWithLogitsLoss (optional pos_weight)
- Auto threshold tuning on validation (optimize F1 by default)
- Save: metrics, predictions, MF plots, 2D slices, 3D pairwise surfaces, confusion matrix, ROC, classification report csv+heatmap
"""

import os
import json
import time
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
    classification_report, f1_score, precision_score, recall_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# User Paths (EDIT IF NEEDED)
# =========================
TRAIN_PATH = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\train.csv"
VAL_PATH   = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\val.csv"
TEST_PATH  = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\test.csv"

# Output base dir (your requested location)
BASE_OUTDIR = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\feature_4\modular_tskfis_model"


# =========================
# Core Config
# =========================
SEED = 42

TARGET_COL = "cardio"
FEATURE_COLS = ["ap_hi", "ap_lo", "age_years", "cholesterol"]

# 2 modules (2 features each)
MODULES = [
    ("Module1_AgeChol", ["age_years", "cholesterol"]),
    ("Module2_BP",      ["ap_hi", "ap_lo"]),
]

# MF numbers (cholesterol is ordinal 1/2/3 -> 3 MFs; others continuous -> 3 MFs)
FEATURE_MF_NUM = {
    "ap_hi": 3,
    "ap_lo": 3,
    "age_years": 3,
    "cholesterol": 3
}

# Training
EPOCHS = 60
BATCH_SIZE = 512
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12

# Threshold tuning
THRESH_SEARCH_MIN = 0.10
THRESH_SEARCH_MAX = 0.90
THRESH_SEARCH_STEPS = 81
THRESH_OBJECTIVE = "f1"   # "f1" (recommended) or "youden" or "balanced_acc"

# Loss weighting
USE_POS_WEIGHT = True     # You can turn off if you are sure balanced and you want clean baseline
AUTO_POS_WEIGHT = True    # If True, compute pos_weight = (#neg/#pos) from train
MANUAL_POS_WEIGHT = 1.0   # Used if AUTO_POS_WEIGHT=False

# Visualizations
SURFACE_GRID_N = 80
SLICE_N = 160


# =========================
# Utilities
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def timestamp_run_dir(base_outdir: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_outdir, f"tskfis_run_weighted_and_threshold{ts}")


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def is_discrete_ordinal(series: pd.Series) -> bool:
    """Heuristic: cholesterol-style ordinal {1,2,3} or {0,1,2} etc."""
    uniq = sorted(series.dropna().unique().tolist())
    if len(uniq) <= 5:
        # Many medical ordinal vars are small set
        return True
    return False


def print_cuda_diagnostics():
    # English comments: diagnosis to explain why GPU not used
    print("---- PyTorch / CUDA Diagnostics ----")
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)  # None => CPU build
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("GPU[0] name:", torch.cuda.get_device_name(0))
    print("-----------------------------------")


def safe_load_state_dict(model: nn.Module, ckpt_path: str, device: torch.device):
    """
    Avoid FutureWarning by using weights_only=True when supported.
    """
    try:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch does not support weights_only
        sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)


# =========================
# TSK Module (per module rule-base)
# =========================
class TSKModule(nn.Module):
    """
    A classic TSK-FIS block:
    - Gaussian MF per feature (mu, sigma)
    - Rule firing strength: product of memberships
    - Normalize firing strengths
    - Consequent (linear) per rule: f_r(x) = w_r^T x + b_r
    - Output: sum( w_norm_r * f_r )
    """
    def __init__(self, feat_names: List[str], feat_index: List[int], mf_nums: List[int], device: torch.device):
        super().__init__()
        self.feat_names = feat_names
        self.feat_index = feat_index
        self.mf_nums = mf_nums
        self.device = device

        self.num_features = len(feat_index)
        self.num_rules = int(np.prod(mf_nums))

        # MF params in Z-space: learnable (mu, log_sigma)
        self.mf_mu = nn.ParameterList()
        self.mf_log_sigma = nn.ParameterList()
        for k in mf_nums:
            self.mf_mu.append(nn.Parameter(torch.zeros(k)))
            self.mf_log_sigma.append(nn.Parameter(torch.zeros(k)))  # sigma=exp(log_sigma)

        # Consequent parameters per rule (R x F) + bias (R)
        self.conseq_w = nn.Parameter(torch.zeros(self.num_rules, self.num_features))
        self.conseq_b = nn.Parameter(torch.zeros(self.num_rules))

        # Precompute rule MF index tuples (like truth table over MFs)
        self.rule_mf_index = self._build_rule_index_tuples(mf_nums)

    @staticmethod
    def _build_rule_index_tuples(mf_nums: List[int]) -> List[Tuple[int, ...]]:
        """
        Example: mf_nums=[3,3] -> 9 rules:
        (0,0),(0,1),(0,2),(1,0),...,(2,2)
        """
        grids = [list(range(m)) for m in mf_nums]
        out = [[]]
        for g in grids:
            out = [prev + [x] for prev in out for x in g]
        return [tuple(x) for x in out]

    def init_from_kmeans(self, Xz_module: np.ndarray):
        """
        KMeans init for each feature MFs in Z-space.
        Xz_module: [N, F_module]
        """
        for j in range(self.num_features):
            k = self.mf_nums[j]
            col = Xz_module[:, j].reshape(-1, 1)

            km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
            km.fit(col)
            centers = np.sort(km.cluster_centers_.reshape(-1))

            # sigma estimate: use within-cluster std; fallback small spread
            labels = km.labels_
            sigmas = []
            for ci in range(k):
                pts = col[labels == ci].reshape(-1)
                if len(pts) >= 2:
                    s = float(np.std(pts) + 1e-3)
                else:
                    s = 0.5
                sigmas.append(s)
            sigmas = np.array(sigmas, dtype=float)

            with torch.no_grad():
                self.mf_mu[j].copy_(torch.tensor(centers, dtype=torch.float32))
                self.mf_log_sigma[j].copy_(torch.log(torch.tensor(sigmas, dtype=torch.float32)))

    def gauss_mf(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # mu,sigma: [K], x:[B] -> [B,K]
        x = x.unsqueeze(1)
        return torch.exp(-0.5 * ((x - mu.unsqueeze(0)) / (sigma.unsqueeze(0) + 1e-8)) ** 2)

    def forward(self, x_module: torch.Tensor) -> torch.Tensor:
        """
        x_module: [B,F_module] in Z-space
        return: logit contribution [B]
        """
        B = x_module.shape[0]

        # MF degrees per feature
        mf_degs = []
        for j in range(self.num_features):
            mu = self.mf_mu[j]
            sigma = torch.exp(self.mf_log_sigma[j])
            deg = self.gauss_mf(x_module[:, j], mu, sigma)  # [B,K]
            mf_degs.append(deg)

        # rule firing strength: product across features -> [B,R]
        w = torch.ones(B, self.num_rules, device=x_module.device)
        for r, mf_idx_tuple in enumerate(self.rule_mf_index):
            prod = 1.0
            for j, mf_idx in enumerate(mf_idx_tuple):
                prod = prod * mf_degs[j][:, mf_idx]
            w[:, r] = prod

        # normalize
        w_sum = torch.sum(w, dim=1, keepdim=True) + 1e-8
        w_norm = w / w_sum

        # consequents
        f = torch.matmul(x_module, self.conseq_w.t()) + self.conseq_b.unsqueeze(0)  # [B,R]

        # weighted sum
        out = torch.sum(w_norm * f, dim=1)  # [B]
        return out


# =========================
# Modular Wrapper
# =========================
class ModularTSKFIS(nn.Module):
    def __init__(self, feature_cols, modules, feature_mf_num, device):
        super().__init__()
        self.feature_cols = feature_cols
        self.device = device

        feat_index_map = {f: i for i, f in enumerate(feature_cols)}
        self.module_names = []
        self.module_feat_names = []
        self.module_models = nn.ModuleList()

        for mod_name, feats in modules:
            feats = [f for f in feats if f in feat_index_map]
            idx = [feat_index_map[f] for f in feats]
            mf_nums = [feature_mf_num[f] for f in feats]
            self.module_names.append(mod_name)
            self.module_feat_names.append(feats)
            self.module_models.append(TSKModule(feats, idx, mf_nums, device))

        # learnable module weights
        self.module_logits = nn.Parameter(torch.zeros(len(self.module_models)))

    def init_mfs(self, X_train_z: np.ndarray):
        for mod in self.module_models:
            idx = mod.feat_index
            Xz = X_train_z[:, idx]
            mod.init_from_kmeans(Xz)

    def forward(self, Xz: torch.Tensor) -> torch.Tensor:
        module_outs = []
        for mod in self.module_models:
            x_m = Xz[:, mod.feat_index]
            module_outs.append(mod(x_m))
        module_outs = torch.stack(module_outs, dim=1)  # [B,M]
        alpha = F.softmax(self.module_logits, dim=0).unsqueeze(0)  # [1,M]
        logit = torch.sum(alpha * module_outs, dim=1)
        return logit

    def module_weights(self) -> Dict[str, float]:
        w = F.softmax(self.module_logits, dim=0).detach().cpu().numpy()
        return {self.module_names[i]: float(w[i]) for i in range(len(w))}


# =========================
# Data
# =========================
def load_split(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame, feature_cols, target_col):
    df = df.copy()
    needed = feature_cols + [target_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df[needed].dropna()

    X = df[feature_cols].astype(float).values
    y = df[target_col].astype(int).values
    return X, y, df


# =========================
# Plots: MF + module weights
# =========================
def plot_module_weights(run_dir, weight_dict):
    plots_dir = ensure_dir(os.path.join(run_dir, "plots_module"))
    names = list(weight_dict.keys())
    vals = [weight_dict[k] for k in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, vals)
    plt.title("Module Weights (softmax)")
    plt.ylabel("Weight")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "module_weights.png"), dpi=160)
    plt.close()


def plot_mf_sets(run_dir, feature_cols, scaler: StandardScaler, model: ModularTSKFIS, X_train_orig: np.ndarray):
    plots_dir = ensure_dir(os.path.join(run_dir, "plots_mf"))

    # Map feature -> (module_index, local_index)
    feat_to_mod = {}
    for mi, mod in enumerate(model.module_models):
        for lj, fname in enumerate(mod.feat_names):
            feat_to_mod[fname] = (mi, lj)

    for f_i, fname in enumerate(feature_cols):
        if fname not in feat_to_mod:
            continue

        mi, lj = feat_to_mod[fname]
        mod = model.module_models[mi]
        with torch.no_grad():
            mu_z = mod.mf_mu[lj].detach().cpu().numpy()
            sig_z = torch.exp(mod.mf_log_sigma[lj]).detach().cpu().numpy()

        # grid in z-space
        xz = (X_train_orig[:, f_i] - scaler.mean_[f_i]) / (scaler.scale_[f_i] + 1e-8)
        xz_min, xz_max = np.percentile(xz, 0.5), np.percentile(xz, 99.5)
        grid_z = np.linspace(xz_min, xz_max, 400)

        # MF(z)
        plt.figure(figsize=(10, 6))
        for k in range(len(mu_z)):
            yk = np.exp(-0.5 * ((grid_z - mu_z[k]) / (sig_z[k] + 1e-8)) ** 2)
            plt.plot(grid_z, yk, linewidth=2)
        plt.title(f"Gaussian MFs in Z-score space: {fname}")
        plt.xlabel(f"{fname} (z-score)")
        plt.ylabel("Membership degree")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"mf_z_{fname}.png"), dpi=160)
        plt.close()

        # transform to original
        mu_o = mu_z * scaler.scale_[f_i] + scaler.mean_[f_i]
        sig_o = sig_z * scaler.scale_[f_i]

        xo_min, xo_max = np.percentile(X_train_orig[:, f_i], 0.5), np.percentile(X_train_orig[:, f_i], 99.5)
        grid_o = np.linspace(xo_min, xo_max, 400)

        plt.figure(figsize=(10, 6))
        for k in range(len(mu_o)):
            yk = np.exp(-0.5 * ((grid_o - mu_o[k]) / (sig_o[k] + 1e-8)) ** 2)
            plt.plot(grid_o, yk, linewidth=2)
        plt.title(f"Gaussian MFs in original units: {fname}")
        plt.xlabel(fname)
        plt.ylabel("Membership degree")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"mf_original_{fname}.png"), dpi=160)
        plt.close()

        # histogram + overlay
        plt.figure(figsize=(10, 6))
        plt.hist(X_train_orig[:, f_i], bins=40, density=True, alpha=0.35)
        for k in range(len(mu_o)):
            yk = np.exp(-0.5 * ((grid_o - mu_o[k]) / (sig_o[k] + 1e-8)) ** 2)
            plt.plot(grid_o, yk / (np.max(yk) + 1e-8), linewidth=2)
        plt.title(f"Histogram + MF overlay (original): {fname}")
        plt.xlabel(fname)
        plt.ylabel("Density / normalized MF")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"hist_overlay_{fname}.png"), dpi=160)
        plt.close()


# =========================
# Response surfaces (2D slices + pairwise 3D)
# =========================
def make_median_point(train_df: pd.DataFrame, feature_cols):
    med = {}
    for f in feature_cols:
        med[f] = float(np.median(train_df[f].values))
    return med


@torch.no_grad()
def predict_prob(model, scaler, X_orig: np.ndarray, device):
    Xz = (X_orig - scaler.mean_) / (scaler.scale_ + 1e-8)
    Xt = torch.tensor(Xz, dtype=torch.float32, device=device)
    logit = model(Xt).detach().cpu().numpy()
    return sigmoid_np(logit)


def plot_2d_slices(run_dir, model, scaler, train_df, feature_cols):
    plots_dir = ensure_dir(os.path.join(run_dir, "plots_2d_slices"))
    med = make_median_point(train_df, feature_cols)

    def slice_values(fname):
        # English comments: for ordinal discrete (cholesterol), use fixed levels
        if fname == "cholesterol":
            return [1.0, 2.0, 3.0]
        vals = train_df[fname].values.astype(float)
        return [float(np.percentile(vals, 25)), float(np.percentile(vals, 50)), float(np.percentile(vals, 75))]

    for x_name in feature_cols:
        for slicer_name in feature_cols:
            if slicer_name == x_name:
                continue

            x_vals = train_df[x_name].values.astype(float)
            x_min, x_max = np.percentile(x_vals, 1), np.percentile(x_vals, 99)
            xs = np.linspace(x_min, x_max, SLICE_N)

            plt.figure(figsize=(10, 6))
            for s_val in slice_values(slicer_name):
                X_grid = np.zeros((len(xs), len(feature_cols)), dtype=float)
                for k, f in enumerate(feature_cols):
                    X_grid[:, k] = med[f]
                X_grid[:, feature_cols.index(x_name)] = xs
                X_grid[:, feature_cols.index(slicer_name)] = s_val

                prob = predict_prob(model, scaler, X_grid, device=model.device)
                plt.plot(xs, prob, linewidth=2, label=f"{slicer_name}={s_val:.3g}")

            plt.title(f"2D Slice Curves: vary {x_name}, slices by {slicer_name} (others=median)")
            plt.xlabel(x_name)
            plt.ylabel("Predicted probability")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"slice_{x_name}_by_{slicer_name}.png"), dpi=160)
            plt.close()


def plot_3d_surfaces_pairwise(run_dir, model, scaler, train_df, feature_cols):
    plots_dir = ensure_dir(os.path.join(run_dir, "plots_3d_surfaces"))
    med = make_median_point(train_df, feature_cols)

    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            pairs.append((feature_cols[i], feature_cols[j]))

    for fx, fy in pairs:
        xvals = train_df[fx].values.astype(float)
        yvals = train_df[fy].values.astype(float)

        xs = np.linspace(np.percentile(xvals, 1), np.percentile(xvals, 99), SURFACE_GRID_N)

        # cholesterol is discrete -> 3 planes (not smooth surface)
        if fy == "cholesterol":
            ys = np.array([1.0, 2.0, 3.0], dtype=float)
        else:
            ys = np.linspace(np.percentile(yvals, 1), np.percentile(yvals, 99), SURFACE_GRID_N)

        XX, YY = np.meshgrid(xs, ys)
        grid = np.zeros((XX.size, len(feature_cols)), dtype=float)
        for k, f in enumerate(feature_cols):
            grid[:, k] = med[f]
        grid[:, feature_cols.index(fx)] = XX.reshape(-1)
        grid[:, feature_cols.index(fy)] = YY.reshape(-1)

        prob = predict_prob(model, scaler, grid, device=model.device)
        ZZ = prob.reshape(YY.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=True, alpha=0.9)
        ax.set_title(f"3D Surface: {fx} vs {fy} (others=median)")
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_zlabel("Prob")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"surface_{fx}_vs_{fy}.png"), dpi=160)
        plt.close()


# =========================
# Metrics + plots
# =========================
def plot_confusion(run_dir, cm, split_name):
    plots_dir = ensure_dir(os.path.join(run_dir, "plots_metrics"))
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix ({split_name})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{split_name}.png"), dpi=160)
    plt.close()


def plot_roc(run_dir, y_true, y_prob, split_name):
    plots_dir = ensure_dir(os.path.join(run_dir, "plots_metrics"))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    plt.title(f"ROC Curve ({split_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"roc_{split_name}.png"), dpi=160)
    plt.close()
    return float(auc)


def classification_report_full(y_true, y_pred):
    """
    Must include:
    precision, recall, f1-score, support, accuracy, micro avg, weighted avg
    """
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # micro avg (single-label classification): equals accuracy when using hard predictions
    rep["micro avg"] = {
        "precision": float(acc),
        "recall": float(acc),
        "f1-score": float(acc),
        "support": int(len(y_true))
    }
    rep["accuracy"] = float(acc)
    return rep


def save_classification_report(run_dir, rep_dict, split_name):
    out_dir = ensure_dir(os.path.join(run_dir, "reports"))

    # CSV
    rows = []
    for key, vals in rep_dict.items():
        if isinstance(vals, dict):
            rows.append({
                "label": key,
                "precision": vals.get("precision", None),
                "recall": vals.get("recall", None),
                "f1-score": vals.get("f1-score", None),
                "support": vals.get("support", None),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"classification_report_{split_name}.csv"), index=False, encoding="utf-8-sig")

    # Heatmap (as you want)
    metrics = ["precision", "recall", "f1-score"]
    labels = ["0", "1", "micro avg", "weighted avg"]
    mat = []
    for lab in labels:
        if lab in rep_dict:
            mat.append([rep_dict[lab].get(m, 0.0) for m in metrics])
        else:
            mat.append([0.0, 0.0, 0.0])
    mat = np.array(mat, dtype=float)

    plt.figure(figsize=(8, 4.8))
    plt.imshow(mat, interpolation="nearest")
    plt.title(f"Classification Report Heatmap ({split_name})")
    plt.xticks(np.arange(len(metrics)), metrics)
    plt.yticks(np.arange(len(labels)), labels)
    plt.colorbar()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.3f}",
                     ha="center", va="center",
                     color="white" if mat[i, j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"classification_report_{split_name}.png"), dpi=160)
    plt.close()


# =========================
# Threshold Tuning
# =========================
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """
    Search threshold on validation.
    Objective:
    - "f1": maximize F1 of class 1
    - "youden": maximize (TPR - FPR)
    - "balanced_acc": maximize (TPR + TNR)/2
    """
    thresholds = np.linspace(THRESH_SEARCH_MIN, THRESH_SEARCH_MAX, THRESH_SEARCH_STEPS)
    best = {
        "objective": THRESH_OBJECTIVE,
        "threshold": 0.5,
        "score": -1e9,
        "precision": None,
        "recall": None,
        "f1": None,
        "tn": None, "fp": None, "fn": None, "tp": None
    }

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_true, pred, zero_division=0)
        rec  = recall_score(y_true, pred, zero_division=0)
        f1   = f1_score(y_true, pred, zero_division=0)

        # compute objective
        if THRESH_OBJECTIVE == "f1":
            score = f1
        elif THRESH_OBJECTIVE == "youden":
            tpr = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)
            score = tpr - fpr
        elif THRESH_OBJECTIVE == "balanced_acc":
            tpr = tp / (tp + fn + 1e-8)
            tnr = tn / (tn + fp + 1e-8)
            score = 0.5 * (tpr + tnr)
        else:
            score = f1

        if score > best["score"]:
            best.update({
                "threshold": float(t),
                "score": float(score),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            })

    return best


# =========================
# Train / Eval
# =========================
def make_loader(Xz, y, batch_size, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xz, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def eval_split(model, scaler, X_orig, y_true, threshold=0.5):
    Xz = (X_orig - scaler.mean_) / (scaler.scale_ + 1e-8)
    Xt = torch.tensor(Xz, dtype=torch.float32, device=model.device)
    logits = model(Xt).detach().cpu().numpy()
    prob = sigmoid_np(logits)
    pred = (prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, prob)
    acc = accuracy_score(y_true, pred)
    return prob, pred, float(auc), float(acc)


def compute_pos_weight(y_train: np.ndarray) -> float:
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    if pos <= 0:
        return 1.0
    return neg / pos


def train_model(run_dir, model, scaler, X_train_orig, y_train, X_val_orig, y_val):
    X_train_z = (X_train_orig - scaler.mean_) / (scaler.scale_ + 1e-8)
    X_val_z   = (X_val_orig   - scaler.mean_) / (scaler.scale_ + 1e-8)

    train_loader = make_loader(X_train_z, y_train, BATCH_SIZE, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # pos_weight
    if USE_POS_WEIGHT:
        if AUTO_POS_WEIGHT:
            pw = compute_pos_weight(y_train)
        else:
            pw = float(MANUAL_POS_WEIGHT)
        pos_weight_t = torch.tensor([pw], dtype=torch.float32, device=model.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        pw = 1.0
        criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_path = os.path.join(run_dir, "best_model.pt")
    no_improve = 0

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(model.device)
            yb = yb.to(model.device)

            opt.zero_grad(set_to_none=True)
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # validation AUC only (threshold will be tuned later)
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_val_z, dtype=torch.float32, device=model.device)
            logits = model(Xt).detach().cpu().numpy()
            prob = sigmoid_np(logits)
            val_auc = float(roc_auc_score(y_val, prob))
            val_pred = (prob >= 0.5).astype(int)
            val_acc = float(accuracy_score(y_val, val_pred))

        ep = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_auc": val_auc,
            "val_acc@0.5": val_acc,
            "pos_weight": float(pw),
            "module_weights": model.module_weights()
        }
        history.append(ep)

        print(f"[Epoch {epoch:03d}] loss={ep['train_loss']:.4f}  val_auc={val_auc:.4f}  val_acc@0.5={val_acc:.4f}  weights={ep['module_weights']}")

        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val AUC={best_val_auc:.4f}")
                break

    save_json(history, os.path.join(run_dir, "train_history.json"))
    return best_path, best_val_auc, float(pw)


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    # GPU selection logic
    print_cuda_diagnostics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = timestamp_run_dir(BASE_OUTDIR)
    ensure_dir(run_dir)

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")

    # Load data
    df_train = load_split(TRAIN_PATH)
    df_val   = load_split(VAL_PATH)
    df_test  = load_split(TEST_PATH)

    X_train, y_train, df_train2 = prepare_data(df_train, FEATURE_COLS, TARGET_COL)
    X_val,   y_val,   df_val2   = prepare_data(df_val,   FEATURE_COLS, TARGET_COL)
    X_test,  y_test,  df_test2  = prepare_data(df_test,  FEATURE_COLS, TARGET_COL)

    # Z-score on train only
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Build model
    model = ModularTSKFIS(
        feature_cols=FEATURE_COLS,
        modules=MODULES,
        feature_mf_num=FEATURE_MF_NUM,
        device=device
    ).to(device)

    # Init MFs by KMeans on train Z-space
    X_train_z = scaler.transform(X_train)
    model.init_mfs(X_train_z)

    # Save config
    config = {
        "features": FEATURE_COLS,
        "modules": MODULES,
        "feature_mf_num": FEATURE_MF_NUM,
        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path": TEST_PATH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "use_pos_weight": USE_POS_WEIGHT,
        "auto_pos_weight": AUTO_POS_WEIGHT,
        "manual_pos_weight": MANUAL_POS_WEIGHT,
        "threshold_search": {
            "min": THRESH_SEARCH_MIN,
            "max": THRESH_SEARCH_MAX,
            "steps": THRESH_SEARCH_STEPS,
            "objective": THRESH_OBJECTIVE
        }
    }
    save_json(config, os.path.join(run_dir, "config.json"))

    # Train
    best_path, best_val_auc, pos_weight_value = train_model(
        run_dir, model, scaler, X_train, y_train, X_val, y_val
    )

    # Load best checkpoint
    safe_load_state_dict(model, best_path, device=device)
    model.eval()

    # Plot module weights
    plot_module_weights(run_dir, model.module_weights())

    # MF plots
    plot_mf_sets(run_dir, FEATURE_COLS, scaler, model, X_train_orig=X_train)

    # 2D slices + 3D surfaces (pairwise)
    plot_2d_slices(run_dir, model, scaler, df_train2, FEATURE_COLS)
    plot_3d_surfaces_pairwise(run_dir, model, scaler, df_train2, FEATURE_COLS)

    # =========
    # Threshold tuning on VAL
    # =========
    val_prob, _, val_auc, _ = eval_split(model, scaler, X_val, y_val, threshold=0.5)
    best_thr_info = find_best_threshold(y_val, val_prob)
    best_thr = best_thr_info["threshold"]

    save_json(best_thr_info, os.path.join(run_dir, "best_threshold.json"))
    print("Best threshold (val):", best_thr_info)

    # =========
    # Evaluate splits using best threshold
    # =========
    out_pred_dir = ensure_dir(os.path.join(run_dir, "predictions"))
    metrics = {
        "best_val_auc": float(best_val_auc),
        "pos_weight_used": float(pos_weight_value),
        "best_threshold": best_thr_info,
        "splits": {}
    }

    for split_name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        prob, pred, auc, acc = eval_split(model, scaler, Xs, ys, threshold=best_thr)

        # Save predictions
        pd.DataFrame({
            "y_true": ys,
            "y_pred_prob": prob,
            "y_pred": pred
        }).to_csv(os.path.join(out_pred_dir, f"{split_name}_pred.csv"), index=False, encoding="utf-8-sig")

        # Confusion + ROC
        cm = confusion_matrix(ys, pred)
        plot_confusion(run_dir, cm, split_name)
        auc_plot = plot_roc(run_dir, ys, prob, split_name)

        # Classification report
        rep = classification_report_full(ys, pred)
        save_classification_report(run_dir, rep, split_name)

        metrics["splits"][split_name] = {
            "auc": float(auc_plot),
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": rep
        }

    # Save overall metrics + scaler
    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    save_json(
        {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "feature_cols": FEATURE_COLS},
        os.path.join(run_dir, "scaler.json")
    )

    print("Done. Outputs saved to:", run_dir)
    print("Note: If Device is still CPU, you likely installed CPU-only PyTorch.")


if __name__ == "__main__":
    main()
