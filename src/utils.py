"""
utils.py
--------
Shared utilities:
  - Evaluation (balanced accuracy, per-class sensitivity, confusion matrix)
  - Training helpers (Caffe InvLR scheduler, model size)
  - Plotting (per-class sensitivity bars, compression waterfall, heatmap)
  - JSON/CSV logging
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from .dataset import CLASS_NAMES, HIGH_RISK_CLASSES


# ── Metrics ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module,
             loader,
             device: torch.device,
             label: str = "") -> dict:
    """
    Evaluate model on a dataloader.

    Returns
    -------
    dict with keys: accuracy, balanced_accuracy, per_class_sensitivity,
                    confusion_matrix, report
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc      = (all_preds == all_labels).mean()
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    cm       = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    report   = classification_report(all_labels, all_preds,
                                     target_names=CLASS_NAMES,
                                     output_dict=True,
                                     zero_division=0)

    # Per-class sensitivity = recall = TP / (TP + FN)
    per_class_sensitivity = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        per_class_sensitivity[cls] = tp / (tp + fn + 1e-9) * 100

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f}")
    for cls in CLASS_NAMES:
        risk_tag = "🔴" if cls in HIGH_RISK_CLASSES else "  "
        print(f"  {risk_tag} {cls}: {per_class_sensitivity[cls]:.1f}%")

    return {
        "accuracy":               float(acc),
        "balanced_accuracy":      float(bal_acc),
        "per_class_sensitivity":  per_class_sensitivity,
        "confusion_matrix":       cm.tolist(),
        "report":                 report,
    }


# ── Model size ─────────────────────────────────────────────────────────────────
def model_size_kb(model: nn.Module) -> float:
    """Estimate model size in KB from parameter count (float32)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1024


# ── Caffe InvLR scheduler ──────────────────────────────────────────────────────
def caffe_inv_lr_scheduler(optimizer,
                            lr0:   float = 1e-3,
                            gamma: float = 1e-4,
                            power: float = 0.75) -> LambdaLR:
    """
    Implements Caffe InvLR policy:
        lr(t) = lr0 * (1 + gamma * t)^(-power)
    stepped per iteration (not per epoch).
    """
    def lr_lambda(iteration):
        return (1 + gamma * iteration) ** (-power)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ── Plotting ───────────────────────────────────────────────────────────────────
STAGE_LABELS = ["Baseline", "After Pruning", "After Quant.", "After Huffman"]
RISK_COLORS  = {"HIGH": "#d62728", "low": "#1f77b4"}


def plot_per_class_sensitivity(results_per_stage: list[dict],
                               model_name: str = "",
                               save_path:  str = None):
    """
    Bar chart of per-class sensitivity at each compression stage.
    Red labels = HIGH clinical risk. Dashed line = 80% threshold.

    Parameters
    ----------
    results_per_stage : list of dicts from evaluate(), one per stage
    """
    n_stages  = len(results_per_stage)
    n_classes = len(CLASS_NAMES)
    x         = np.arange(n_classes)
    width     = 0.8 / n_stages
    colors    = plt.cm.Blues(np.linspace(0.3, 0.9, n_stages))

    fig, ax = plt.subplots(figsize=(12, 5))

    for si, (res, col) in enumerate(zip(results_per_stage, colors)):
        sens   = [res["per_class_sensitivity"][c] for c in CLASS_NAMES]
        offset = (si - n_stages / 2 + 0.5) * width
        ax.bar(x + offset, sens, width, label=STAGE_LABELS[si], color=col)

    ax.axhline(80, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="80% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n({'HIGH' if c in HIGH_RISK_CLASSES else 'low'})" for c in CLASS_NAMES],
        fontsize=9,
    )
    # Colour HIGH-risk labels red
    for label, cls in zip(ax.get_xticklabels(), CLASS_NAMES):
        if cls in HIGH_RISK_CLASSES:
            label.set_color("red")
            label.set_fontweight("bold")

    ax.set_ylabel("Sensitivity (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{model_name} Per-Class Sensitivity at Each Compression Stage")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_compression_waterfall(sizes_kb:   list[float],
                               model_name: str  = "",
                               save_path:  str  = None):
    """Bar chart (log scale) showing size reduction across stages."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(STAGE_LABELS[:len(sizes_kb)], sizes_kb,
                  color=["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"])
    ax.set_yscale("log")
    ax.set_ylabel("Size (KB, log scale)")
    ax.set_title(f"{model_name} Compression Pipeline")

    for bar, size in zip(bars, sizes_kb):
        label = f"{size/1024:.1f} MB" if size > 1024 else f"{size:.0f} KB"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.1, label,
                ha="center", va="bottom", fontsize=9)

    ratio = sizes_kb[0] / sizes_kb[-1]
    ax.set_title(f"{model_name} Compression: {sizes_kb[0]/1024:.1f} MB → "
                 f"{sizes_kb[-1]/1024:.1f} MB  ({ratio:.1f}×)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_sensitivity_heatmap(baseline_res: dict,
                             final_res:    dict,
                             model_name:   str  = "",
                             save_path:    str  = None):
    """
    Heatmap of Δ sensitivity (final − baseline) per class.
    Green = improvement, Red = degradation.
    """
    delta = {
        cls: final_res["per_class_sensitivity"][cls]
             - baseline_res["per_class_sensitivity"][cls]
        for cls in CLASS_NAMES
    }

    df    = pd.DataFrame(delta, index=["Δ Sensitivity (pp)"]).T
    annot = df.applymap(lambda v: f"{v:+.1f}")

    fig, ax = plt.subplots(figsize=(10, 2.5))
    sns.heatmap(df.T, annot=annot.T, fmt="", center=0,
                cmap="RdYlGn", linewidths=0.5, ax=ax,
                cbar_kws={"label": "pp change"})

    # Bold HIGH-risk x-tick labels
    for label in ax.get_xticklabels():
        if label.get_text() in HIGH_RISK_CLASSES:
            label.set_fontweight("bold")
            label.set_color("darkred")

    ax.set_title(f"{model_name} Sensitivity Δ vs Baseline — "
                 "Negative = Clinically Dangerous | Bold red = HIGH RISK")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_compression_accuracy_tradeoff(sizes_kb:      list[float],
                                       bal_accs:      list[float],
                                       mel_sens:      list[float],
                                       model_name:    str  = "",
                                       save_path:     str  = None):
    """Line plot: compression ratio × balanced accuracy & melanoma sensitivity."""
    ratios = [sizes_kb[0] / s for s in sizes_kb]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(ratios, [b * 100 for b in bal_accs],
             "b-o", label="Balanced Accuracy (%)")
    ax2.plot(ratios, mel_sens, "r--s", label="Melanoma Sensitivity (%)")
    ax2.axhline(80, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax1.set_xlabel("Compression Ratio (×)")
    ax1.set_ylabel("Balanced Accuracy (%)", color="blue")
    ax2.set_ylabel("Melanoma Sensitivity (%)", color="red")
    ax1.set_title(f"{model_name}: Compression Ratio vs. Accuracy & Melanoma Sensitivity")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

    for i, (r, stage) in enumerate(zip(ratios, STAGE_LABELS[:len(ratios)])):
        ax1.annotate(stage, (r, bal_accs[i] * 100),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Logging ────────────────────────────────────────────────────────────────────
def save_metrics(metrics: dict, path: str):
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[utils] Saved metrics → {path}")


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def append_training_log(log_path: str, row: dict):
    """Append one row to a CSV training log (creates file if missing)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(log_path):
        df_new.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(log_path, index=False)
