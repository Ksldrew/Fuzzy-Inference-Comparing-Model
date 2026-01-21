
# ======================================================
# One-file GPU Modular TSK-FIS for Heart Disease Prediction
# Training + Evaluation + Full Visualizations + White-box Exports
#
# Author: (your name)
# Notes:
# - This script implements the canonical 5-layer TSK-FIS computation inside each module:
#   L1 Fuzzification (Gaussian MFs) -> L2 Rule firing -> L3 Normalization ->
#   L4 TSK consequent -> L5 Weighted aggregation
# - A modular "sum-of-subsystems" architecture is used: final_logit = Σ_k (alpha_k * f_k(x))
# - BCEWithLogitsLoss is used for binary label "cardio" (0/1).
#
# Usage (Windows example):
#   python tskfis_onefile_with_visuals.py ^
#     --train "C:\path\train.csv" --val "C:\path\val.csv" --test "C:\path\test.csv" ^
#     --out  "C:\path\outputs"
#
# ======================================================

import argparse
import itertools
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


# ------------------------------
# 0) Utilities
# ------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------
# 1) Dataset
# ------------------------------

class HeartDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------
# 2) MF initialization helpers
# ------------------------------

def make_mfs_for_feature(num_mf: int):
    """
    Construct initial (centers, sigmas) in standardized space (z-score domain).
    """
    if num_mf == 2:
        centers = [-1.0, 1.0]
        sigmas = [0.7, 0.7]
    elif num_mf == 3:
        centers = [-1.5, 0.0, 1.5]
        sigmas = [0.8, 0.8, 0.8]
    elif num_mf == 4:
        centers = [-2.0, -0.7, 0.7, 2.0]
        sigmas = [0.9, 0.7, 0.7, 0.9]
    elif num_mf == 5:
        centers = [-2.2, -1.1, 0.0, 1.1, 2.2]
        sigmas = [0.9, 0.8, 0.8, 0.8, 0.9]
    else:
        # fallback
        centers = [-1.5, 0.0, 1.5]
        sigmas = [0.8, 0.8, 0.8]
    return centers, sigmas


# ------------------------------
# 3) TSK-FIS module definitions (the "true architecture")
# ------------------------------

class GaussianMF(nn.Module):
    """
    Layer 1: Fuzzification using trainable Gaussian membership functions.
    μ(x) = exp(-(x-μ)^2 / (2σ^2))
    """
    def __init__(self, centers, sigmas, device):
        super().__init__()
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32, device=device))
        self.sigmas  = nn.Parameter(torch.tensor(sigmas,  dtype=torch.float32, device=device))

    def forward(self, x):  # x: (B,1)
        diff = x - self.centers.view(1, -1)  # (B,M)
        denom = 2.0 * (self.sigmas.view(1, -1) ** 2) + 1e-8
        return torch.exp(-(diff ** 2) / denom)  # (B,M)


class TSKModule(nn.Module):
    """
    A sub-TSK-FIS subsystem operating on a subset of input features.

    Implements canonical 5-layer TSK-FIS:
      L1: Gaussian MF for each feature -> memberships
      L2: Rule firing strength = product of chosen MFs
      L3: Normalize firing strengths
      L4: Consequent per rule: f_r(x)=b_r + Σ w_rj * x_j
      L5: Output f(x)=Σ (normalized_firing_r * f_r(x))
    """
    def __init__(self, feature_indices, mf_params, device):
        super().__init__()
        self.feature_indices = feature_indices
        self.num_features = len(feature_indices)
        self.device = device

        self.mf_layers = nn.ModuleList()
        self.num_mfs_per_feature = []
        for centers, sigmas in mf_params:
            self.mf_layers.append(GaussianMF(centers, sigmas, device))
            self.num_mfs_per_feature.append(len(centers))

        combos = list(itertools.product(*[range(m) for m in self.num_mfs_per_feature]))
        self.num_rules = len(combos)

        self.register_buffer("rule_mf_indices", torch.tensor(combos, dtype=torch.long, device=device))

        # (R, F+1): [bias, coeffs...]
        self.consequents = nn.Parameter(torch.randn(self.num_rules, self.num_features + 1, device=device) * 0.01)

    def forward(self, x):  # x: (B, D)
        x_sub = x[:, self.feature_indices]  # (B,F)
        B = x_sub.size(0)
        device = x_sub.device

        # L1: memberships per feature
        mf_vals = []
        for j in range(self.num_features):
            mf_vals.append(self.mf_layers[j](x_sub[:, j:j+1]))  # (B,Mj)

        # L2: rule firing strengths
        firing = torch.ones(B, self.num_rules, device=device)
        for j in range(self.num_features):
            firing *= mf_vals[j][:, self.rule_mf_indices[:, j]]

        # L3: normalization
        firing_sum = firing.sum(dim=1, keepdim=True) + 1e-8
        norm_firing = firing / firing_sum

        # L4: rule consequents
        ones = torch.ones(B, 1, device=device)
        x_ext = torch.cat([ones, x_sub], dim=1)  # (B,F+1)
        rule_outputs = torch.matmul(x_ext, self.consequents.t())  # (B,R)

        # L5: aggregation
        return (norm_firing * rule_outputs).sum(dim=1)  # (B,)

    @torch.no_grad()
    def explain(self, x):
        """
        Export intermediate values for white-box analysis (Step 3–5 style).
        """
        x_sub = x[:, self.feature_indices]
        B = x_sub.size(0)
        device = x_sub.device

        mf_vals = []
        for j in range(self.num_features):
            mf_vals.append(self.mf_layers[j](x_sub[:, j:j+1]))

        firing = torch.ones(B, self.num_rules, device=device)
        for j in range(self.num_features):
            firing *= mf_vals[j][:, self.rule_mf_indices[:, j]]

        firing_sum = firing.sum(dim=1, keepdim=True) + 1e-8
        norm_firing = firing / firing_sum

        ones = torch.ones(B, 1, device=device)
        x_ext = torch.cat([ones, x_sub], dim=1)
        rule_outputs = torch.matmul(x_ext, self.consequents.t())

        module_output = (norm_firing * rule_outputs).sum(dim=1)

        return {
            "x_sub": x_sub.detach().cpu().numpy(),
            "mf_vals": [mv.detach().cpu().numpy() for mv in mf_vals],
            "firing": firing.detach().cpu().numpy(),
            "norm_firing": norm_firing.detach().cpu().numpy(),
            "rule_outputs": rule_outputs.detach().cpu().numpy(),
            "module_output": module_output.detach().cpu().numpy()
        }


class ModularTSKFIS(nn.Module):
    """
    Full modular architecture:
      final_logit = Σ_k (alpha_k * f_k(x))

    Note:
    - alpha_k are learnable and not constrained to sum to 1.
      If you want "probability-like" module weights, you can softmax them.
    """
    def __init__(self, modules, device, weight_mode: str = "raw"):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.module_weights = nn.Parameter(torch.ones(len(modules), device=device))
        self.weight_mode = weight_mode  # "raw" or "softmax"
        self.device = device

    def forward(self, x, return_all: bool = False):
        module_outputs = torch.stack([m(x) for m in self.modules_list], dim=1)  # (B,M)

        if self.weight_mode == "softmax":
            w = torch.softmax(self.module_weights, dim=0)  # (M,)
        else:
            w = self.module_weights  # (M,)

        weighted = module_outputs * w.view(1, -1)  # (B,M)
        final_logit = weighted.sum(dim=1)          # (B,)

        if return_all:
            return final_logit, module_outputs, weighted, w
        return final_logit


# ------------------------------
# 4) Training / eval
# ------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * yb.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_model(model, loader, criterion, device, return_probs=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        logits = model(Xb)
        loss = criterion(logits, yb)

        total_loss += float(loss.item()) * yb.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

        if return_probs:
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    if return_probs:
        return avg_loss, acc, np.concatenate(all_labels), np.concatenate(all_probs)
    return avg_loss, acc


@torch.no_grad()
def collect_module_outputs(model, loader, device):
    model.eval()
    final_logits_list, module_raw_list, module_weighted_list = [], [], []
    weights_snapshot = None

    for Xb, _ in loader:
        Xb = Xb.to(device)
        final_logit, module_raw, module_weighted, w = model(Xb, return_all=True)
        final_logits_list.append(final_logit.detach().cpu().numpy())
        module_raw_list.append(module_raw.detach().cpu().numpy())
        module_weighted_list.append(module_weighted.detach().cpu().numpy())
        weights_snapshot = w.detach().cpu().numpy()

    return (
        np.concatenate(final_logits_list, axis=0),
        np.concatenate(module_raw_list, axis=0),
        np.concatenate(module_weighted_list, axis=0),
        weights_snapshot
    )


# ------------------------------
# 5) Visualizations
# ------------------------------

def plot_training_curves(history, out_path):
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(df["epoch"], df["train_acc"], label="Train Acc")
    ax1.plot(df["epoch"], df["val_acc"], label="Val Acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["train_loss"], linestyle="--", label="Train Loss")
    ax2.plot(df["epoch"], df["val_loss"], linestyle="--", label="Val Loss")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")

    plt.title("Training & Validation Accuracy / Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return auc


def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])

    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm


def save_mf_params_json(modules, module_feature_names, out_path):
    mf_dict = {}
    for module, feat_names in zip(modules, module_feature_names):
        for feat_name, mf_layer in zip(feat_names, module.mf_layers):
            centers = mf_layer.centers.detach().cpu().numpy().tolist()
            sigmas = mf_layer.sigmas.detach().cpu().numpy().tolist()
            mf_dict[feat_name] = {"num_mf": len(centers), "centers": centers, "sigmas": sigmas}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mf_dict, f, indent=2)


def plot_mfs_standardized(mf_json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(mf_json_path, "r", encoding="utf-8") as f:
        mf_data = json.load(f)

    x = np.linspace(-3, 3, 400)
    for feature, params in mf_data.items():
        centers, sigmas = params["centers"], params["sigmas"]
        plt.figure(figsize=(6, 4))
        for c, s in zip(centers, sigmas):
            y = np.exp(-((x - c) ** 2) / (2 * (s ** 2 + 1e-8)))
            plt.plot(x, y, label=f"c={c:.2f}, σ={s:.2f}")
        plt.title(f"MFs for {feature} (z-score)")
        plt.xlabel("z-score")
        plt.ylabel("Membership")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mf_std_{feature}.png"), dpi=150)
        plt.close()


def plot_mfs_original_units(mf_json_path, scaler_path, feature_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(mf_json_path, "r", encoding="utf-8") as f:
        mf_data = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    means, scales = scaler.mean_, scaler.scale_
    for feature, params in mf_data.items():
        if feature not in feature_cols:
            continue
        idx = feature_cols.index(feature)
        c_std = np.array(params["centers"])
        s_std = np.array(params["sigmas"])
        c_real = c_std * scales[idx] + means[idx]
        s_real = s_std * scales[idx]

        x_min = means[idx] - 3 * scales[idx]
        x_max = means[idx] + 3 * scales[idx]
        x = np.linspace(x_min, x_max, 400)

        plt.figure(figsize=(6, 4))
        for c, s in zip(c_real, s_real):
            y = np.exp(-((x - c) ** 2) / (2 * (s ** 2 + 1e-8)))
            plt.plot(x, y, label=f"μ={c:.2f}, σ={s:.2f}")
        plt.title(f"MFs for {feature} (original units)")
        plt.xlabel(feature)
        plt.ylabel("Membership")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mf_real_{feature}.png"), dpi=150)
        plt.close()


def plot_hist_with_mf_overlay(df_all, feature_cols, mf_json_path, scaler_path, out_dir, bins=30):
    os.makedirs(out_dir, exist_ok=True)
    with open(mf_json_path, "r", encoding="utf-8") as f:
        mf_data = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    means, scales = scaler.mean_, scaler.scale_

    for feature in feature_cols:
        if feature not in mf_data:
            continue
        idx = feature_cols.index(feature)
        centers_std = np.array(mf_data[feature]["centers"])
        sigmas_std = np.array(mf_data[feature]["sigmas"])
        centers_real = centers_std * scales[idx] + means[idx]
        sigmas_real = sigmas_std * scales[idx]

        vals = df_all[feature].dropna().values
        if len(vals) < 2:
            continue

        x = np.linspace(vals.min(), vals.max(), 400)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.hist(vals, bins=bins, density=True, alpha=0.35)
        ax1.set_xlabel(feature)
        ax1.set_ylabel("Density")
        ax2 = ax1.twinx()
        for c, s in zip(centers_real, sigmas_real):
            y = np.exp(-((x - c) ** 2) / (2 * (s ** 2 + 1e-8)))
            ax2.plot(x, y, label=f"μ={c:.2f}, σ={s:.2f}")
        ax2.set_ylabel("Membership")
        ax2.legend(loc="upper right", fontsize=8)
        plt.title(f"Histogram + Learned MFs: {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"overlay_{feature}.png"), dpi=150)
        plt.close()


@torch.no_grad()
def sensitivity_analysis_1d(model, scaler, feature_cols, feature_name, out_path, num_points=60, device=None):
    if feature_name not in feature_cols:
        return
    if device is None:
        device = next(model.parameters()).device

    idx = feature_cols.index(feature_name)
    means, scales = scaler.mean_, scaler.scale_

    base = means.copy()
    x_min = means[idx] - 2.5 * scales[idx]
    x_max = means[idx] + 2.5 * scales[idx]
    xs = np.linspace(x_min, x_max, num_points)

    probs = []
    for v in xs:
        sample = base.copy()
        sample[idx] = v
        sample_scaled = scaler.transform(sample.reshape(1, -1))
        x_tensor = torch.tensor(sample_scaled, dtype=torch.float32, device=device)
        prob = torch.sigmoid(model(x_tensor)).item()
        probs.append(prob)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, probs, marker="o", markersize=3)
    plt.xlabel(feature_name)
    plt.ylabel("Predicted Risk Probability")
    plt.grid(True, alpha=0.3)
    plt.title(f"Sensitivity: {feature_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_module_weights(weights, module_names, out_path):
    w = np.array(weights).astype(float)
    w_abs = np.abs(w)
    w_norm = w_abs / (w_abs.sum() + 1e-9)

    labels = module_names if len(module_names) == len(w_norm) else [f"Module {i+1}" for i in range(len(w_norm))]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, w_norm)
    plt.title("Module Importance (normalized |weight|)")
    plt.ylabel("Relative Contribution")
    plt.ylim(0, max(w_norm) * 1.25 if len(w_norm) else 1)

    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y + 0.01, f"{y*100:.1f}%", ha="center")

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_module_scatter(pred_df, num_modules, out_dir, label_col="cardio"):
    os.makedirs(out_dir, exist_ok=True)
    n = len(pred_df)
    x_idx = np.arange(n)
    labels = pred_df[label_col].values

    for k in range(num_modules):
        col = f"module_{k+1}_weighted"
        if col not in pred_df.columns:
            continue

        y_vals = pred_df[col].values
        plt.figure(figsize=(10, 4.8))
        mask0 = labels == 0
        mask1 = labels == 1
        plt.scatter(x_idx[mask0], y_vals[mask0], alpha=0.6, label="cardio=0")
        plt.scatter(x_idx[mask1], y_vals[mask1], alpha=0.6, label="cardio=1")
        plt.xlabel("Sample index (Excel row = index + 2)")
        plt.ylabel(f"Module {k+1} weighted output")
        plt.title(f"Module {k+1} contribution scatter")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"module_{k+1}_scatter.png"), dpi=150)
        plt.close()


def export_step1_to_5(model: ModularTSKFIS,
                      modules: list,
                      module_feature_names: list,
                      feature_cols: list,
                      X_raw: np.ndarray,
                      X_scaled: np.ndarray,
                      y_true: np.ndarray,
                      pred_df: pd.DataFrame,
                      out_dir: str,
                      device,
                      num_samples: int = 5,
                      label_col="cardio"):
    """
    Export detailed intermediate values (Step 1–5) for correctly predicted samples.
    """
    os.makedirs(out_dir, exist_ok=True)

    correct_mask = pred_df[label_col].values.astype(int) == pred_df["prediction"].values.astype(int)
    idxs = np.where(correct_mask)[0]
    if len(idxs) == 0:
        return

    sel = idxs[:min(num_samples, len(idxs))]

    for idx in sel:
        excel_row = int(idx) + 2
        x_raw = X_raw[idx].tolist()
        x_scaled = X_scaled[idx].tolist()

        entry = {
            "sample_index": int(idx),
            "excel_row": excel_row,
            "true_label": int(y_true[idx]),
            "predicted_label": int(pred_df.loc[idx, "prediction"]),
            "predicted_probability": float(pred_df.loc[idx, "probability"]),
            "step1_raw_features": {feature_cols[i]: float(x_raw[i]) for i in range(len(feature_cols))},
            "step2_normalized_features": {feature_cols[i]: float(x_scaled[i]) for i in range(len(feature_cols))},
            "modules": []
        }

        x_tensor = torch.tensor(X_scaled[idx:idx+1], dtype=torch.float32, device=device)

        for m_idx, (m, fnames) in enumerate(zip(modules, module_feature_names), start=1):
            expl = m.explain(x_tensor)
            # expl arrays are numpy already
            x_sub = expl["x_sub"][0].tolist()
            mf_vals = [mv[0].tolist() for mv in expl["mf_vals"]]
            firing = expl["firing"][0].tolist()
            norm_firing = expl["norm_firing"][0].tolist()
            rule_outputs = expl["rule_outputs"][0].tolist()
            module_output = float(expl["module_output"][0])

            entry["modules"].append({
                "module_index": m_idx,
                "features_used": {fnames[j]: float(x_sub[j]) for j in range(len(fnames))},
                "step3_membership_mu": {fnames[j]: mf_vals[j] for j in range(len(fnames))},
                "step4_rule_firing_raw": firing,
                "step4_rule_firing_normalized": norm_firing,
                "step5_rule_outputs": rule_outputs,
                "step5_module_output_f": module_output
            })

        out_path = os.path.join(out_dir, f"val_sample_step1to5_idx{idx}_row{excel_row}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)


# ------------------------------
# 6) Main pipeline
# ------------------------------

@dataclass
class ModuleSpec:
    name: str
    features: list


def build_default_module_specs(feature_cols):
    """
    Default module grouping (matches your FYP idea).
    Only uses features that exist in current CSV.
    """
    specs = [
        ModuleSpec("Bio-Demographics", ["age_years", "gender", "height", "weight"]),
        ModuleSpec("Blood Pressure", ["ap_hi", "ap_lo"]),
        ModuleSpec("Metabolic", ["cholesterol", "gluc"]),
        ModuleSpec("Lifestyle", ["smoke", "alco", "active"]),
    ]
    # filter missing
    out = []
    for s in specs:
        feats = [f for f in s.features if f in feature_cols]
        if feats:
            out.append(ModuleSpec(s.name, feats))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--val", required=True, help="Path to val.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--label", default="cardio", help="Label column name (0/1)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_mode", choices=["raw", "softmax"], default="raw",
                        help="Module weight mode: raw (unconstrained) or softmax (sums to 1).")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"tskfis_run_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df_train = pd.read_csv(args.train)
    df_val = pd.read_csv(args.val)
    df_test = pd.read_csv(args.test)

    # Choose features (keep order)
    candidate_cols = [
        "age_years", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active"
    ]
    feature_cols = [c for c in candidate_cols if c in df_train.columns]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Check CSV headers.")

    # MF counts: customize here if you want
    custom_mf_num = {
        "gender": 2, "smoke": 2, "alco": 2, "active": 2,
        "age_years": 5, "height": 4, "weight": 4,
        "ap_hi": 5, "ap_lo": 5,
        "cholesterol": 3, "gluc": 3,
    }
    feature_mf_num = {c: custom_mf_num.get(c, 3) for c in feature_cols}

    # Arrays
    X_train_raw = df_train[feature_cols].values.astype("float32")
    y_train = df_train[args.label].values.astype("float32")
    X_val_raw = df_val[feature_cols].values.astype("float32")
    y_val = df_val[args.label].values.astype("float32")
    X_test_raw = df_test[feature_cols].values.astype("float32")
    y_test = df_test[args.label].values.astype("float32")

    # Save all DF for overlays
    df_all = pd.concat([df_train, df_val, df_test], axis=0)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # DataLoaders
    train_loader = DataLoader(HeartDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(HeartDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(HeartDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build modules
    feature_index_map = {name: i for i, name in enumerate(feature_cols)}
    module_specs = build_default_module_specs(feature_cols)

    modules, module_feature_names, module_names = [], [], []
    for spec in module_specs:
        idxs = [feature_index_map[f] for f in spec.features]
        mfs = [make_mfs_for_feature(feature_mf_num[f]) for f in spec.features]
        modules.append(TSKModule(idxs, mfs, device))
        module_feature_names.append(spec.features)
        module_names.append(spec.name)

    model = ModularTSKFIS(modules, device=device, weight_mode=args.weight_mode).to(device)

    # loss: dynamic pos_weight
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight_value = (n_neg / n_pos) if n_pos > 0 else 1.0

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop with early stopping
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            break

    # save log + curves
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
    plot_training_curves(history, os.path.join(out_dir, "training_curves.png"))

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    # Save MF params and plot
    mf_json_path = os.path.join(out_dir, "mf_params.json")
    save_mf_params_json(modules, module_feature_names, mf_json_path)
    plot_mfs_standardized(mf_json_path, os.path.join(out_dir, "mfs_zscore"))
    plot_mfs_original_units(mf_json_path, os.path.join(out_dir, "scaler.pkl"), feature_cols, os.path.join(out_dir, "mfs_original"))
    plot_hist_with_mf_overlay(df_all, feature_cols, mf_json_path, os.path.join(out_dir, "scaler.pkl"), os.path.join(out_dir, "mf_overlay"))

    # Evaluate on test set (AUC, confusion matrix)
    test_loss, test_acc, y_true, y_prob = eval_model(model, test_loader, criterion, device, return_probs=True)
    auc = plot_roc(y_true, y_prob, os.path.join(out_dir, "roc_test.png"))

    y_pred = (y_prob > 0.5).astype(int)
    cm = plot_confusion(y_true.astype(int), y_pred, os.path.join(out_dir, "confusion_matrix.png"))

    # Export transparent outputs for TEST
    final_logits, module_raw, module_weighted, w_snapshot = collect_module_outputs(model, test_loader, device)
    prob_from_logits = sigmoid_np(final_logits)
    pred_from_logits = (prob_from_logits > 0.5).astype(int)

    pred_df = pd.DataFrame(X_test_raw, columns=feature_cols)
    pred_df[args.label] = y_test.astype(int)
    pred_df["final_logit"] = final_logits
    pred_df["probability"] = prob_from_logits
    pred_df["prediction"] = pred_from_logits

    num_modules = module_raw.shape[1]
    for i in range(num_modules):
        pred_df[f"module_{i+1}_raw"] = module_raw[:, i]
        pred_df[f"module_{i+1}_weighted"] = module_weighted[:, i]

    abs_weighted = np.abs(module_weighted)
    pct = abs_weighted / (abs_weighted.sum(axis=1, keepdims=True) + 1e-8)
    for i in range(num_modules):
        pred_df[f"module_{i+1}_pct"] = pct[:, i]

    pred_df.to_csv(os.path.join(out_dir, "test_predictions_whitebox.csv"), index=False)
    plot_module_scatter(pred_df, num_modules, os.path.join(out_dir, "module_scatter_test"), label_col=args.label)

    # module weight plot
    plot_module_weights(w_snapshot, module_names, os.path.join(out_dir, "module_weights.png"))

    # sensitivity for each feature
    for fname in feature_cols:
        sensitivity_analysis_1d(model, scaler, feature_cols, fname, os.path.join(out_dir, f"sens_{fname}.png"), device=device)

    # Validation white-box export + step1-5 detail for a few correct samples
    val_logits, val_module_raw, val_module_weighted, _ = collect_module_outputs(model, val_loader, device)
    val_prob = sigmoid_np(val_logits)
    val_pred = (val_prob > 0.5).astype(int)

    val_df = pd.DataFrame(X_val_raw, columns=feature_cols)
    val_df[args.label] = y_val.astype(int)
    val_df["final_logit"] = val_logits
    val_df["probability"] = val_prob
    val_df["prediction"] = val_pred

    for i in range(num_modules):
        val_df[f"module_{i+1}_raw"] = val_module_raw[:, i]
        val_df[f"module_{i+1}_weighted"] = val_module_weighted[:, i]
    val_abs = np.abs(val_module_weighted)
    val_pct = val_abs / (val_abs.sum(axis=1, keepdims=True) + 1e-8)
    for i in range(num_modules):
        val_df[f"module_{i+1}_pct"] = val_pct[:, i]

    val_df.to_csv(os.path.join(out_dir, "val_predictions_whitebox.csv"), index=False)
    plot_module_scatter(val_df, num_modules, os.path.join(out_dir, "module_scatter_val"), label_col=args.label)

    export_step1_to_5(model, modules, module_feature_names, feature_cols,
                      X_val_raw, X_val, y_val.astype(int), val_df,
                      os.path.join(out_dir, "step1to5_val_explain"),
                      device=device, num_samples=5, label_col=args.label)

    # Save a concise metrics summary (for your Chapter 4)
    metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "pos_weight_used": float(pos_weight_value),
        "features": feature_cols,
        "modules": [{ "name": n, "features": f } for n, f in zip(module_names, module_feature_names)],
        "weight_mode": args.weight_mode,
        "module_weights_snapshot": w_snapshot.tolist() if w_snapshot is not None else None,
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
