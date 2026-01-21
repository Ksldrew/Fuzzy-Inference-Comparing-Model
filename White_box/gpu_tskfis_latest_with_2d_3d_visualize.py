# ======================================================
# GPU-based Modular TSK-FIS for Heart Disease Prediction
# (Training + Full Visualization + Transparent Module Outputs + Excel-Friendly Contributions)
#
# - Adds: per-module 3D surfaces + 2D slices (saved in same folder as 3D)
# - Adds: final combined model 3D + 2D slices (this IS the combined visualization)
# - Fixes: absolute file paths (no os.path.join(base_dir, "C:/...") misuse)
# ======================================================

# ======================================================
# 0. Imports
# ======================================================

import itertools
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from tqdm import tqdm

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})


# ======================================================
# 1. DATASET & MF CONFIG
# ======================================================

class HeartDataset(Dataset):
    """PyTorch Dataset for heart disease data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def infer_num_mf_for_feature(series):
    """Decide number of MFs based on unique values in the original dataset column."""
    uniq = sorted(series.dropna().unique().tolist())
    if len(uniq) == 2 and (uniq == [0, 1] or uniq == [1, 2]):
        return 2
    if len(uniq) == 3 and (uniq == [0, 1, 2] or uniq == [1, 2, 3]):
        return 3
    return 3  # default


def make_mfs_for_feature(num_mf: int):
    """
    Construct initial (centers, sigmas) in standardized space (~ -3 to +3).
    """
    if num_mf == 2:
        centers = [-1.0, 1.0]
        sigmas = [0.7, 0.7]
    elif num_mf == 3:
        centers = [-1.5, 0.0, 1.5]
        sigmas = [0.8, 0.8, 0.8]
    else:
        # fallback (you can expand if you want 4/5 MF different init)
        centers = [-2.0, -1.0, 0.0, 1.0, 2.0][:num_mf]
        sigmas = [0.8] * num_mf
    return centers, sigmas


# ======================================================
# 2. TSK-FIS MODULE DEFINITIONS
# ======================================================

class GaussianMF(nn.Module):
    """Gaussian MF layer for one feature."""
    def __init__(self, centers, sigmas, device):
        super().__init__()
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32, device=device))
        self.sigmas  = nn.Parameter(torch.tensor(sigmas,  dtype=torch.float32, device=device))

    def forward(self, x):
        diff = x - self.centers.view(1, -1)
        denom = 2.0 * (self.sigmas.view(1, -1) ** 2) + 1e-8
        return torch.exp(-(diff ** 2) / denom)


class TSKModule(nn.Module):
    """One TSK module using a subset of features."""
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

        self.register_buffer(
            "rule_mf_indices",
            torch.tensor(combos, dtype=torch.long, device=device)
        )

        self.consequents = nn.Parameter(
            torch.randn(self.num_rules, self.num_features + 1, device=device) * 0.01
        )

    def forward(self, x):
        """Return module output f_k(x)."""
        x_sub = x[:, self.feature_indices]  # (B, F)
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

        return (norm_firing * rule_outputs).sum(dim=1)

    def explain(self, x):
        """Return Step 3–5 internals for white-box analysis."""
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
            "x_sub": x_sub,
            "mf_vals": mf_vals,
            "firing": firing,
            "norm_firing": norm_firing,
            "rule_outputs": rule_outputs,
            "module_output": module_output
        }


class ModularTSKFIS(nn.Module):
    """
    Full modular TSK-FIS:
      final_logit = sum_k (w_k * module_k_raw)
    """
    def __init__(self, modules, device):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.module_weights = nn.Parameter(torch.ones(len(modules), device=device))
        self.device = device

    def forward(self, x, return_all: bool = False):
        module_outputs = torch.stack([m(x) for m in self.modules_list], dim=1)  # (B, M)
        weighted = module_outputs * self.module_weights.view(1, -1)            # (B, M)
        final_logit = weighted.sum(dim=1)                                      # (B,)

        if return_all:
            return final_logit, module_outputs, weighted
        return final_logit


# ======================================================
# 3. TRAINING & EVALUATION
# ======================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for X_batch, y_batch in loop:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        loop.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.4f}"})

    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, criterion, device, return_probs=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * y_batch.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        if return_probs:
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total

    if return_probs:
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return avg_loss, acc, all_labels, all_probs
    return avg_loss, acc


@torch.no_grad()
def collect_module_outputs(model, loader, device):
    model.eval()
    final_logits_list = []
    module_raw_list = []
    module_weighted_list = []

    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        final_logit, module_raw, module_weighted = model(X_batch, return_all=True)
        final_logits_list.append(final_logit.cpu().numpy())
        module_raw_list.append(module_raw.cpu().numpy())
        module_weighted_list.append(module_weighted.cpu().numpy())

    final_logits = np.concatenate(final_logits_list, axis=0)
    module_raw = np.concatenate(module_raw_list, axis=0)
    module_weighted = np.concatenate(module_weighted_list, axis=0)

    return final_logits, module_raw, module_weighted


@torch.no_grad()
def export_step1_to_5_for_val_samples(
    model,
    modules,
    module_feature_names,
    feature_cols,
    X_val_raw,
    X_val,
    y_val,
    val_pred_df,
    device,
    out_dir,
    num_samples=5
):
    os.makedirs(out_dir, exist_ok=True)

    correct_mask = val_pred_df["cardio"].values == val_pred_df["prediction"].values
    correct_idx = np.where(correct_mask)[0]

    if len(correct_idx) == 0:
        print("[VAL-STEP1-5] No correctly predicted samples, skip exporting.")
        return

    selected_idx = correct_idx[: min(num_samples, len(correct_idx))]
    print(f"[VAL-STEP1-5] Exporting Step 1–5 details for indices: {selected_idx.tolist()}")

    for idx in selected_idx:
        excel_row = int(idx) + 2
        y_true = int(y_val[idx])
        y_pred = int(val_pred_df.loc[idx, "prediction"])
        prob = float(val_pred_df.loc[idx, "probability"])

        raw_feat = X_val_raw[idx].tolist()
        norm_feat = X_val[idx].tolist()

        sample_dict = {
            "sample_index": int(idx),
            "excel_row": excel_row,
            "true_label": y_true,
            "predicted_label": y_pred,
            "predicted_probability": prob,
            "step1_raw_features": {name: float(raw_feat[i]) for i, name in enumerate(feature_cols)},
            "step2_normalized_features": {name: float(norm_feat[i]) for i, name in enumerate(feature_cols)},
            "modules": []
        }

        x_tensor = torch.tensor(X_val[idx:idx+1], dtype=torch.float32, device=device)

        for m_idx, (module, feat_names) in enumerate(zip(modules, module_feature_names), start=1):
            expl = module.explain(x_tensor)

            x_sub = expl["x_sub"].detach().cpu().numpy()[0].tolist()
            mf_vals = [mv.detach().cpu().numpy()[0].tolist() for mv in expl["mf_vals"]]
            firing = expl["firing"].detach().cpu().numpy()[0].tolist()
            norm_firing = expl["norm_firing"].detach().cpu().numpy()[0].tolist()
            rule_outputs = expl["rule_outputs"].detach().cpu().numpy()[0].tolist()
            module_output = float(expl["module_output"].detach().cpu().numpy()[0])

            module_entry = {
                "module_index": m_idx,
                "module_name": f"Module_{m_idx}",
                "features_used": {name: float(x_sub[j]) for j, name in enumerate(feat_names)},
                "step3_membership_mu": {feat_name: mf_vals[j] for j, feat_name in enumerate(feat_names)},
                "step4_rule_firing_raw": firing,
                "step4_rule_firing_normalized": norm_firing,
                "step5_rule_outputs": rule_outputs,
                "step5_module_output_f": module_output
            }

            sample_dict["modules"].append(module_entry)

        out_path = os.path.join(out_dir, f"val_sample_step1to5_idx{idx}_row{excel_row}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample_dict, f, indent=2, ensure_ascii=False)

        print(f"[VAL-STEP1-5] Saved explanation for sample idx={idx} (Excel row={excel_row})")


# ======================================================
# 4. PLOTTING & VISUALIZATION
# ======================================================

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
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_mf_params_json(modules, module_feature_names, out_path):
    mf_dict = {}
    for module, feat_names in zip(modules, module_feature_names):
        for feat_name, mf_layer in zip(feat_names, module.mf_layers):
            centers = mf_layer.centers.detach().cpu().numpy().tolist()
            sigmas  = mf_layer.sigmas.detach().cpu().numpy().tolist()
            mf_dict[feat_name] = {"num_mf": len(centers), "centers": centers, "sigmas": sigmas}
    with open(out_path, "w") as f:
        json.dump(mf_dict, f, indent=2)


def plot_mfs_standardized(mf_json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(mf_json_path, "r") as f:
        mf_data = json.load(f)

    x = np.linspace(-3, 3, 400)
    for feature, params in mf_data.items():
        centers, sigmas = params["centers"], params["sigmas"]
        plt.figure(figsize=(6, 4))
        for c, s in zip(centers, sigmas):
            y = np.exp(-(x - c)**2 / (2 * (s**2)))
            plt.plot(x, y, label=f"c={c:.2f}, σ={s:.2f}")
        plt.title(f"MFs for {feature} (Standardized)")
        plt.xlabel("z-score")
        plt.ylabel("Membership")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"step4_mf_std_{feature}.png"))
        plt.close()


def plot_mfs_original_units(mf_json_path, scaler_path, feature_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(mf_json_path, "r") as f:
        mf_data = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    means, scales = scaler.mean_, scaler.scale_

    for feature, params in mf_data.items():
        if feature not in feature_cols:
            continue
        idx = feature_cols.index(feature)
        c_std, s_std = np.array(params["centers"]), np.array(params["sigmas"])
        c_real = c_std * scales[idx] + means[idx]
        s_real = s_std * scales[idx]
        x_min, x_max = means[idx] - 3 * scales[idx], means[idx] + 3 * scales[idx]
        x = np.linspace(x_min, x_max, 400)

        plt.figure(figsize=(6, 4))
        for c, s in zip(c_real, s_real):
            y = np.exp(-(x - c)**2 / (2 * (s**2 + 1e-8)))
            plt.plot(x, y, label=f"μ={c:.2f}, σ={s:.2f}")

        plt.title(f"MFs for {feature} (Original Units)")
        plt.xlabel(feature)
        plt.ylabel("Membership")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"step4_mf_orig_{feature}.png"))
        plt.close()


def plot_histogram_with_mf_overlay(df, feature_cols, mf_json_path, scaler_path, out_dir, bins=30):
    os.makedirs(out_dir, exist_ok=True)
    with open(mf_json_path, "r") as f:
        mf_data = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    means, scales = scaler.mean_, scaler.scale_

    for feature in feature_cols:
        if feature not in mf_data:
            continue
        idx = feature_cols.index(feature)
        c_std, s_std = np.array(mf_data[feature]["centers"]), np.array(mf_data[feature]["sigmas"])
        c_real = c_std * scales[idx] + means[idx]
        s_real = s_std * scales[idx]

        col_values = df[feature].dropna().values
        x_min, x_max = col_values.min(), col_values.max()
        margin = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        x = np.linspace(x_min - margin, x_max + margin, 400)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.hist(col_values, bins=bins, density=True, alpha=0.4)
        ax1.set_xlabel(f"{feature} (original units)")
        ax1.set_ylabel("Density")

        ax2 = ax1.twinx()
        for c, s in zip(c_real, s_real):
            y = np.exp(-(x - c)**2 / (2 * (s**2 + 1e-8)))
            ax2.plot(x, y, label=f"μ={c:.2f}, σ={s:.2f}")
        ax2.set_ylabel("Membership")
        ax2.legend(loc="upper right")

        plt.title(f"Data + Learned MFs for {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"step5_overlay_{feature}.png"))
        plt.close()


@torch.no_grad()
def sensitivity_analysis_1d(model, scaler, feature_cols, feature_name,
                            fixed_values=None, num_points=50, out_path=None, device=None):
    if feature_name not in feature_cols:
        return
    if device is None:
        device = next(model.parameters()).device

    idx = feature_cols.index(feature_name)
    means, scales = scaler.mean_, scaler.scale_
    base = means.copy()

    if fixed_values:
        for fname, val in fixed_values.items():
            if fname in feature_cols:
                base[feature_cols.index(fname)] = val

    x_min, x_max = means[idx] - 2 * scales[idx], means[idx] + 2 * scales[idx]
    xs = np.linspace(x_min, x_max, num_points)
    probs = []

    for val in xs:
        sample = base.copy()
        sample[idx] = val
        sample_scaled = scaler.transform(sample.reshape(1, -1))
        x_tensor = torch.tensor(sample_scaled, dtype=torch.float32, device=device)
        probs.append(torch.sigmoid(model(x_tensor)).item())

    plt.figure(figsize=(6, 4))
    plt.plot(xs, probs, marker="o")
    plt.xlabel(feature_name)
    plt.ylabel("Predicted Risk")
    plt.grid(True)
    plt.title(f"Sensitivity: {feature_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ======================================================
# 5. NEW: 3D SURFACE + 2D SLICES (FINAL + PER MODULE)
# ======================================================

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def plot_surface_and_slices_same_folder(
    feature_x,
    feature_y,
    predict_logit_fn,       # callable(z_input_np)-> float logit
    scaler,
    feature_cols,
    out_dir,
    title,
    grid_n=45,
    slice_z_values=(-1.5, 0.0, 1.5)
):
    """
    Save in SAME folder:
      - 3D surface: 3D_{feature_x}_vs_{feature_y}.png
      - 2D slices:
          slice_fix_{feature_y}_plot_{feature_x}.png
          slice_fix_{feature_x}_plot_{feature_y}.png
    Note:
      Y-axis is always Risk Probability = sigmoid(logit).
    """
    if feature_x not in feature_cols or feature_y not in feature_cols:
        print(f"[SKIP] {title}: missing features.")
        return

    os.makedirs(out_dir, exist_ok=True)

    idx_x = feature_cols.index(feature_x)
    idx_y = feature_cols.index(feature_y)

    mx, sx = scaler.mean_[idx_x], scaler.scale_[idx_x]
    my, sy = scaler.mean_[idx_y], scaler.scale_[idx_y]

    # grid in z space
    x_z = np.linspace(-2.5, 2.5, grid_n)
    y_z = np.linspace(-2.5, 2.5, grid_n)
    Xz, Yz = np.meshgrid(x_z, y_z)
    P = np.zeros_like(Xz, dtype=np.float32)

    base_z = np.zeros((1, len(feature_cols)), dtype=np.float32)

    for i in range(Xz.shape[0]):
        for j in range(Xz.shape[1]):
            sample_z = base_z.copy()
            sample_z[0, idx_x] = Xz[i, j]
            sample_z[0, idx_y] = Yz[i, j]
            logit = float(predict_logit_fn(sample_z))
            P[i, j] = _sigmoid(logit)

    X_real = Xz * sx + mx
    Y_real = Yz * sy + my

    # ---------- 3D surface ----------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_real, Y_real, P, edgecolor='none', alpha=0.9)
    ax.set_xlabel(feature_x, fontsize=11, labelpad=10)
    ax.set_ylabel(feature_y, fontsize=11, labelpad=10)
    ax.set_zlabel("Risk Probability", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14)
    fig.colorbar(surf, shrink=0.5, aspect=6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"3D_{feature_x}_vs_{feature_y}.png"), dpi=140)
    plt.close()

    # ---------- 2D Slice 1: fix Y, plot P vs X ----------
    x_real_line = x_z * sx + mx
    plt.figure(figsize=(7, 5))
    for y_fixed_z in slice_z_values:
        probs = []
        base = np.zeros((1, len(feature_cols)), dtype=np.float32)
        base[0, idx_y] = y_fixed_z
        for xv in x_z:
            sample = base.copy()
            sample[0, idx_x] = xv
            probs.append(_sigmoid(float(predict_logit_fn(sample))))
        y_fixed_real = y_fixed_z * sy + my
        plt.plot(x_real_line, probs, marker="o", linewidth=1, label=f"{feature_y}≈{y_fixed_real:.2f}")
    plt.xlabel(feature_x)
    plt.ylabel("Risk Probability")
    plt.title(f"2D Slices: P vs {feature_x} (fix {feature_y})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"slice_fix_{feature_y}_plot_{feature_x}.png"), dpi=140)
    plt.close()

    # ---------- 2D Slice 2: fix X, plot P vs Y ----------
    y_real_line = y_z * sy + my
    plt.figure(figsize=(7, 5))
    for x_fixed_z in slice_z_values:
        probs = []
        base = np.zeros((1, len(feature_cols)), dtype=np.float32)
        base[0, idx_x] = x_fixed_z
        for yv in y_z:
            sample = base.copy()
            sample[0, idx_y] = yv
            probs.append(_sigmoid(float(predict_logit_fn(sample))))
        x_fixed_real = x_fixed_z * sx + mx
        plt.plot(y_real_line, probs, marker="o", linewidth=1, label=f"{feature_x}≈{x_fixed_real:.2f}")
    plt.xlabel(feature_y)
    plt.ylabel("Risk Probability")
    plt.title(f"2D Slices: P vs {feature_y} (fix {feature_x})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"slice_fix_{feature_x}_plot_{feature_y}.png"), dpi=140)
    plt.close()


def plot_module_weights(model, out_dir):
    weights = model.module_weights.detach().cpu().numpy()
    weights_abs = np.abs(weights)
    weights_norm = weights_abs / (weights_abs.sum() + 1e-9)

    labels = ["Bio-Demographics\n(Age, Gender, Height, Weight)",
              "Blood Pressure\n(Sys, Dia)",
              "Metabolic\n(Chol, Gluc)",
              "Lifestyle\n(Smoke, Alco, Active)"]

    if len(weights_norm) != len(labels):
        labels = [f"Module {i+1}" for i in range(len(weights_norm))]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, weights_norm)
    plt.title("Importance of Each Clinical Module (Learned Weights)", fontsize=14)
    plt.ylabel("Relative Contribution (Normalized)", fontsize=12)
    plt.ylim(0, max(weights_norm)*1.25)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 yval + 0.01,
                 f"{yval*100:.1f}%",
                 ha='center',
                 fontweight='bold')

    out_file = os.path.join(out_dir, "step8_Module_Importance.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    plt.close()


def plot_patient_radar(feature_cols, scaler, out_dir):
    display_labels = ['Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Glucose']
    feat_keys = ['age_years', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']

    valid_indices = []
    valid_labels = []
    for k, lbl in zip(feat_keys, display_labels):
        if k in feature_cols:
            valid_indices.append(feature_cols.index(k))
            valid_labels.append(lbl)

    if not valid_indices:
        return

    avg_vals = scaler.mean_[valid_indices]

    patient_vals = avg_vals.copy()
    # Example profile (for visualization only)
    keys_present = [k for k in feat_keys if k in feature_cols]
    for pos, key in enumerate(keys_present):
        if key == 'age_years':
            patient_vals[pos] = 65.0
        elif key == 'ap_hi':
            patient_vals[pos] = 160.0
        elif key == 'ap_lo':
            patient_vals[pos] = 100.0
        elif key == 'cholesterol':
            patient_vals[pos] = 3.0

    values_norm = patient_vals / (avg_vals + 1e-6)
    avg_ref = np.ones(len(valid_labels))

    angles = np.linspace(0, 2*np.pi, len(valid_labels), endpoint=False).tolist()
    values_norm = np.concatenate((values_norm, [values_norm[0]]))
    avg_ref = np.concatenate((avg_ref, [avg_ref[0]]))
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values_norm, linewidth=2, label='High Risk Example')
    ax.fill(angles, values_norm, alpha=0.25)
    ax.plot(angles, avg_ref, linestyle='--', label='Population Avg')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_labels, fontsize=10)
    ax.set_title("Patient Risk Profile Analysis", y=1.08, fontsize=14)
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0.1))

    out_file = os.path.join(out_dir, "step9_Patient_Explanation_Radar.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    plt.close()


def plot_module_scatter(pred_df, out_dir, num_modules):
    os.makedirs(out_dir, exist_ok=True)
    n_samples = len(pred_df)

    for k in range(num_modules):
        col_w = f"module_{k+1}_weighted"
        if col_w not in pred_df.columns:
            continue

        x_idx = np.arange(n_samples)
        y_vals = pred_df[col_w].values
        labels = pred_df["cardio"].values

        plt.figure(figsize=(10, 5))
        mask0 = (labels == 0)
        mask1 = (labels == 1)
        plt.scatter(x_idx[mask0], y_vals[mask0], alpha=0.6, label="cardio=0")
        plt.scatter(x_idx[mask1], y_vals[mask1], alpha=0.6, label="cardio=1")

        plt.xlabel("Sample Index (0-based; Excel row = index + 2)")
        plt.ylabel(f"Module {k+1} Weighted Output")
        plt.title(f"Module {k+1} Weighted Contribution vs Sample Index")
        plt.grid(True)
        plt.legend()

        abs_vals = np.abs(y_vals)
        top_idx = np.argsort(abs_vals)[-10:]
        for idx in top_idx:
            excel_row = int(idx) + 2
            plt.annotate(str(excel_row),
                         (x_idx[idx], y_vals[idx]),
                         textcoords="offset points",
                         xytext=(0, 5),
                         ha='center',
                         fontsize=8)

        out_file = os.path.join(out_dir, f"step3_module_{k+1}_weighted_scatter.png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=120)
        plt.close()

        print(f"[INFO] Module {k+1} top-10 absolute contributions at CSV rows (Excel row index): {sorted(int(i)+2 for i in top_idx)}")


# ======================================================
# 6. MAIN PIPELINE
# ======================================================

def main():
    # ---------- [USER PATHS] ----------
    train_path = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Preprocessing_Edited_Dataset\train.csv"
    val_path   = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Preprocessing_Edited_Dataset\val.csv"
    test_path  = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\Preprocessing_Edited_Dataset\test.csv"
    # ---------------------------------

    label_col = "cardio"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    root_out_dir = r"C:\Users\asus\Desktop\FYP Improvement\FYP2\White_box"
    os.makedirs(root_out_dir, exist_ok=True)

    output_dir = os.path.join(root_out_dir, f"tskfiswithfullwhitebox_preprocessing_2d_3d_surface{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output Dir: {output_dir}")

    # ---------- Step 1: Load ----------
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}")

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_test  = pd.read_csv(test_path)

    feature_cols = [
        "age_years", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active"
    ]
    feature_cols = [c for c in feature_cols if c in df_train.columns]
    print(f"Using features: {feature_cols}")

    # ---------- MF counts ----------
    custom_mf_num = {
        "gender": 2, "smoke": 2, "alco": 2, "active": 2,
        "age_years": 5, "height": 4, "weight": 4, "ap_hi": 5, "ap_lo": 5,
        "cholesterol": 3, "gluc": 3,
    }

    feature_mf_num = {}
    for col in feature_cols:
        if col in custom_mf_num:
            feature_mf_num[col] = custom_mf_num[col]
        else:
            feature_mf_num[col] = infer_num_mf_for_feature(df_train[col])

    print("MF per feature:", feature_mf_num)

    # ---------- Extract arrays ----------
    X_train = df_train[feature_cols].values.astype("float32")
    y_train = df_train[label_col].values.astype("float32")

    X_val = df_val[feature_cols].values.astype("float32")
    y_val = df_val[label_col].values.astype("float32")

    X_test = df_test[feature_cols].values.astype("float32")
    y_test = df_test[label_col].values.astype("float32")

    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    df_all = pd.concat([df_train, df_val, df_test], axis=0)

    # ---------- Normalize ----------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # DataLoaders
    batch_size = 128
    train_loader = DataLoader(HeartDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(HeartDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(HeartDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # ---------- Build modules ----------
    feature_index_map = {name: idx for idx, name in enumerate(feature_cols)}

    m1_feats = [f for f in ["age_years", "gender", "height", "weight"] if f in feature_cols]
    m2_feats = [f for f in ["ap_hi", "ap_lo"] if f in feature_cols]
    m3_feats = [f for f in ["cholesterol", "gluc"] if f in feature_cols]
    m4_feats = [f for f in ["smoke", "alco", "active"] if f in feature_cols]

    modules = []
    module_feature_names = []
    for feats in [m1_feats, m2_feats, m3_feats, m4_feats]:
        if not feats:
            continue
        idx = [feature_index_map[f] for f in feats]
        mfs = [make_mfs_for_feature(feature_mf_num[f]) for f in feats]
        modules.append(TSKModule(idx, mfs, device))
        module_feature_names.append(feats)

    model = ModularTSKFIS(modules, device).to(device)

    # ---------- Step 2: Train ----------
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight_value = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    print(f"Computed pos_weight = {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4
    )

    num_epochs = 120
    history = []

    best_val_acc = 0.0
    best_state = None

    patience = 30
    epochs_no_improve = 0
    min_delta = 1e-5

    print("Starting Training...")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f" → New best val_acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f" → No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\nEarly Stopping Triggered at epoch {epoch}.")
            print(f"Best Validation Accuracy: {best_val_acc:.4f}")
            break

    pd.DataFrame(history).to_csv(os.path.join(output_dir, "step2_training_log.csv"), index=False)
    plot_training_curves(history, os.path.join(output_dir, "step2_training_acc_loss.png"))

    if best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        print("\nSaved Best Model.")

    # ---------- Save MF params ----------
    mf_json_path = os.path.join(output_dir, "mf_params.json")
    save_mf_params_json(modules, module_feature_names, out_path=mf_json_path)

    # ---------- Step 3: Final Test ----------
    test_loss, test_acc, y_true, y_prob = eval_model(model, test_loader, criterion, device, return_probs=True)
    auc_value = roc_auc_score(y_true, y_prob)

    print(f"\nFINAL TEST ACC: {test_acc:.4f} | AUC: {auc_value:.4f} | Test Loss: {test_loss:.4f}")
    plot_roc_curve(y_true, y_prob, os.path.join(output_dir, "step3_test_roc_curve.png"))

    # ---------- Transparent outputs (TEST) ----------
    print("Collecting per-module outputs for TEST set (transparent Excel analysis)...")
    final_logits, module_raw, module_weighted = collect_module_outputs(model, test_loader, device)

    prob_from_logits = 1.0 / (1.0 + np.exp(-final_logits))
    preds_from_logits = (prob_from_logits > 0.5).astype(int)

    pred_df = pd.DataFrame(X_test_raw, columns=feature_cols)
    pred_df[label_col] = y_true
    pred_df["final_logit"] = final_logits
    pred_df["probability"] = prob_from_logits
    pred_df["prediction"] = preds_from_logits

    num_modules = module_raw.shape[1]
    for i in range(num_modules):
        pred_df[f"module_{i+1}_raw"] = module_raw[:, i]
        pred_df[f"module_{i+1}_weighted"] = module_weighted[:, i]

    abs_weighted = np.abs(module_weighted)
    sum_abs = abs_weighted.sum(axis=1, keepdims=True) + 1e-8
    pct = abs_weighted / sum_abs
    for i in range(num_modules):
        pred_df[f"module_{i+1}_pct"] = pct[:, i]

    csv_path = os.path.join(output_dir, "step3_test_predictions.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"[TEST] Transparent predictions with modules saved to: {csv_path}")

    # ---------- Module summary (TEST) ----------
    summary_rows = []
    for i in range(num_modules):
        w_col = f"module_{i+1}_weighted"
        p_col = f"module_{i+1}_pct"

        mean_weighted_all = pred_df[w_col].mean()
        mean_pct_all = pred_df[p_col].mean()

        mask_pos = (pred_df[label_col] == 1)
        mask_neg = (pred_df[label_col] == 0)
        mean_weighted_pos = pred_df.loc[mask_pos, w_col].mean()
        mean_weighted_neg = pred_df.loc[mask_neg, w_col].mean()
        mean_pct_pos = pred_df.loc[mask_pos, p_col].mean()
        mean_pct_neg = pred_df.loc[mask_neg, p_col].mean()

        summary_rows.append({
            "module_index": i+1,
            "module_name": f"Module_{i+1}",
            "mean_weighted_all": mean_weighted_all,
            "mean_pct_all": mean_pct_all,
            "mean_weighted_pos(cardio=1)": mean_weighted_pos,
            "mean_weighted_neg(cardio=0)": mean_weighted_neg,
            "mean_pct_pos(cardio=1)": mean_pct_pos,
            "mean_pct_neg(cardio=0)": mean_pct_neg
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "step3_module_contribution_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[TEST] Module contribution summary saved to: {summary_path}")

    scatter_dir = os.path.join(output_dir, "step3_module_scatter")
    plot_module_scatter(pred_df, scatter_dir, num_modules)

    # ---------- Transparent outputs (VAL) ----------
    print("Collecting per-module outputs for VALIDATION set (transparent Excel analysis)...")
    val_final_logits, val_module_raw, val_module_weighted = collect_module_outputs(model, val_loader, device)

    val_prob_from_logits = 1.0 / (1.0 + np.exp(-val_final_logits))
    val_preds_from_logits = (val_prob_from_logits > 0.5).astype(int)

    val_pred_df = pd.DataFrame(X_val_raw, columns=feature_cols)
    val_pred_df[label_col] = y_val
    val_pred_df["final_logit"] = val_final_logits
    val_pred_df["probability"] = val_prob_from_logits
    val_pred_df["prediction"] = val_preds_from_logits

    val_num_modules = val_module_raw.shape[1]
    for i in range(val_num_modules):
        val_pred_df[f"module_{i+1}_raw"] = val_module_raw[:, i]
        val_pred_df[f"module_{i+1}_weighted"] = val_module_weighted[:, i]

    val_abs_weighted = np.abs(val_module_weighted)
    val_sum_abs = val_abs_weighted.sum(axis=1, keepdims=True) + 1e-8
    val_pct = val_abs_weighted / val_sum_abs
    for i in range(val_num_modules):
        val_pred_df[f"module_{i+1}_pct"] = val_pct[:, i]

    val_csv_path = os.path.join(output_dir, "step3_val_predictions.csv")
    val_pred_df.to_csv(val_csv_path, index=False)
    print(f"[VAL] Transparent predictions with modules saved to: {val_csv_path}")

    # VAL summary
    val_summary_rows = []
    for i in range(val_num_modules):
        w_col = f"module_{i+1}_weighted"
        p_col = f"module_{i+1}_pct"

        mean_weighted_all = val_pred_df[w_col].mean()
        mean_pct_all = val_pred_df[p_col].mean()

        mask_pos = (val_pred_df[label_col] == 1)
        mask_neg = (val_pred_df[label_col] == 0)
        mean_weighted_pos = val_pred_df.loc[mask_pos, w_col].mean()
        mean_weighted_neg = val_pred_df.loc[mask_neg, w_col].mean()
        mean_pct_pos = val_pred_df.loc[mask_pos, p_col].mean()
        mean_pct_neg = val_pred_df.loc[mask_neg, p_col].mean()

        val_summary_rows.append({
            "module_index": i+1,
            "module_name": f"Module_{i+1}",
            "mean_weighted_all": mean_weighted_all,
            "mean_pct_all": mean_pct_all,
            "mean_weighted_pos(cardio=1)": mean_weighted_pos,
            "mean_weighted_neg(cardio=0)": mean_weighted_neg,
            "mean_pct_pos(cardio=1)": mean_pct_pos,
            "mean_pct_neg(cardio=0)": mean_pct_neg
        })

    val_summary_df = pd.DataFrame(val_summary_rows)
    val_summary_path = os.path.join(output_dir, "step3_val_module_contribution_summary.csv")
    val_summary_df.to_csv(val_summary_path, index=False)
    print(f"[VAL] Module contribution summary saved to: {val_summary_path}")

    val_scatter_dir = os.path.join(output_dir, "step3_val_module_scatter")
    plot_module_scatter(val_pred_df, val_scatter_dir, val_num_modules)

    # Export Step1-5 (VAL)
    explain_dir = os.path.join(output_dir, "step1to5_val_explain")
    export_step1_to_5_for_val_samples(
        model, modules, module_feature_names, feature_cols,
        X_val_raw, X_val, y_val, val_pred_df, device,
        explain_dir, num_samples=5
    )

    # ---------- Step 4/5 MF Visualizations ----------
    print("Generating Standard Plots (MFs)...")
    plot_mfs_standardized(mf_json_path, os.path.join(output_dir, "step4_mf_std"))
    plot_mfs_original_units(
        mf_json_path,
        os.path.join(output_dir, "scaler.pkl"),
        feature_cols,
        os.path.join(output_dir, "step4_mf_original")
    )

    print("Generating Data + MF Overlay Histograms...")
    plot_histogram_with_mf_overlay(
        df_all, feature_cols, mf_json_path,
        os.path.join(output_dir, "scaler.pkl"),
        os.path.join(output_dir, "step5_mf_overlay")
    )

    # ---------- Step 6 sensitivity ----------
    print("Generating 1D Sensitivity Curves...")
    for fname in feature_cols:
        sensitivity_analysis_1d(
            model, scaler, feature_cols, fname,
            out_path=os.path.join(output_dir, f"step6_sens_{fname}.png"),
            device=device
        )

    # ======================================================
    # Step 7 (NEW): FINAL + PER MODULE 3D SURFACES + 2D SLICES
    # ======================================================
    print("Generating Advanced Visualizations (FINAL + Module Surfaces & Slices)...")

    step7_root = os.path.join(output_dir, "step7_surfaces")
    os.makedirs(step7_root, exist_ok=True)

    # ----- FINAL combined model (THIS is the combined visualization) -----
    def final_logit_fn(z_np):
        x = torch.tensor(z_np, dtype=torch.float32, device=device)
        logit = model(x)  # final logit
        return float(logit.detach().cpu().numpy()[0])

    final_dir = os.path.join(step7_root, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    # Choose key pairs for final model (you can add more pairs if you want)
    if "age_years" in feature_cols and "ap_hi" in feature_cols:
        plot_surface_and_slices_same_folder(
            "age_years", "ap_hi",
            final_logit_fn, scaler, feature_cols,
            out_dir=os.path.join(final_dir, "age_years_vs_ap_hi"),
            title="FINAL Combined Model: Risk Probability Surface (age_years vs ap_hi)"
        )

    if "ap_hi" in feature_cols and "ap_lo" in feature_cols:
        plot_surface_and_slices_same_folder(
            "ap_hi", "ap_lo",
            final_logit_fn, scaler, feature_cols,
            out_dir=os.path.join(final_dir, "ap_hi_vs_ap_lo"),
            title="FINAL Combined Model: Risk Probability Surface (ap_hi vs ap_lo)"
        )

    if "cholesterol" in feature_cols and "gluc" in feature_cols:
        plot_surface_and_slices_same_folder(
            "cholesterol", "gluc",
            final_logit_fn, scaler, feature_cols,
            out_dir=os.path.join(final_dir, "cholesterol_vs_gluc"),
            title="FINAL Combined Model: Risk Probability Surface (cholesterol vs gluc)"
        )

    # ----- Per-module surfaces (weighted contribution logit) -----
    modules_dir = os.path.join(step7_root, "modules")
    os.makedirs(modules_dir, exist_ok=True)

    module_names = ["Module1_Bio", "Module2_BP", "Module3_Metabolic", "Module4_Lifestyle"]
    if len(module_names) != len(model.modules_list):
        module_names = [f"Module{i+1}" for i in range(len(model.modules_list))]

    for k in range(len(model.modules_list)):
        mod_name = module_names[k]
        mod_folder = os.path.join(modules_dir, mod_name)
        os.makedirs(mod_folder, exist_ok=True)

        # module weighted contribution logit = w_k * raw_k
        def module_logit_fn(z_np, kk=k):
            x = torch.tensor(z_np, dtype=torch.float32, device=device)
            raw_k = model.modules_list[kk](x)     # (1,)
            w_k = model.module_weights[kk]        # scalar
            logit_k = raw_k * w_k
            return float(logit_k.detach().cpu().numpy()[0])

        # Decide pairs per module
        # Bio: age_years vs weight (or age_years vs height)
        if mod_name.startswith("Module1"):
            if "age_years" in feature_cols and "weight" in feature_cols:
                plot_surface_and_slices_same_folder(
                    "age_years", "weight",
                    module_logit_fn, scaler, feature_cols,
                    out_dir=os.path.join(mod_folder, "age_years_vs_weight"),
                    title=f"{mod_name}: Weighted Contribution Surface (age_years vs weight)"
                )
            elif "age_years" in feature_cols and "height" in feature_cols:
                plot_surface_and_slices_same_folder(
                    "age_years", "height",
                    module_logit_fn, scaler, feature_cols,
                    out_dir=os.path.join(mod_folder, "age_years_vs_height"),
                    title=f"{mod_name}: Weighted Contribution Surface (age_years vs height)"
                )

        # BP: ap_hi vs ap_lo
        if mod_name.startswith("Module2"):
            if "ap_hi" in feature_cols and "ap_lo" in feature_cols:
                plot_surface_and_slices_same_folder(
                    "ap_hi", "ap_lo",
                    module_logit_fn, scaler, feature_cols,
                    out_dir=os.path.join(mod_folder, "ap_hi_vs_ap_lo"),
                    title=f"{mod_name}: Weighted Contribution Surface (ap_hi vs ap_lo)"
                )

        # Metabolic: cholesterol vs gluc
        if mod_name.startswith("Module3"):
            if "cholesterol" in feature_cols and "gluc" in feature_cols:
                plot_surface_and_slices_same_folder(
                    "cholesterol", "gluc",
                    module_logit_fn, scaler, feature_cols,
                    out_dir=os.path.join(mod_folder, "cholesterol_vs_gluc"),
                    title=f"{mod_name}: Weighted Contribution Surface (cholesterol vs gluc)"
                )

        # Lifestyle: smoke vs active (or alco vs active)
        if mod_name.startswith("Module4"):
            if "smoke" in feature_cols and "active" in feature_cols:
                plot_surface_and_slices_same_folder(
                    "smoke", "active",
                    module_logit_fn, scaler, feature_cols,
                    out_dir=os.path.join(mod_folder, "smoke_vs_active"),
                    title=f"{mod_name}: Weighted Contribution Surface (smoke vs active)"
                )
            elif "alco" in feature_cols and "active" in feature_cols:
                plot_surface_and_slices_same_folder(
                    "alco", "active",
                    module_logit_fn, scaler, feature_cols,
                    out_dir=os.path.join(mod_folder, "alco_vs_active"),
                    title=f"{mod_name}: Weighted Contribution Surface (alco vs active)"
                )

    # Step 8/9 visuals
    if len(modules) > 0:
        plot_module_weights(model, output_dir)
    plot_patient_radar(feature_cols, scaler, output_dir)

    print(f"Done! All results saved in {output_dir}")


if __name__ == "__main__":
    print("DEBUG: Starting gpu_tskfis_latest_edit_optimizer.py")
    main()
