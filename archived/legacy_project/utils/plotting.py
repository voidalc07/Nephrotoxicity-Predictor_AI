from __future__ import annotations

import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float, out_path: str) -> None:
    preds = (prob_pos >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_true, preds, labels=[0, 1])
    _ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, prob_pos: np.ndarray, out_path: str) -> float:
    fpr, tpr, _ = metrics.roc_curve(y_true, prob_pos)
    auc = metrics.roc_auc_score(y_true, prob_pos)
    _ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title("ROC Curve")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return auc


def plot_pr(y_true: np.ndarray, prob_pos: np.ndarray, out_path: str) -> float:
    precision, recall, _ = metrics.precision_recall_curve(y_true, prob_pos)
    ap = metrics.average_precision_score(y_true, prob_pos)
    _ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return ap


def plot_coverage_table(coverage_df, out_path: str, metric: str = "f1") -> None:
    # coverage_df columns: coverage_threshold, coverage, metrics...
    _ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(5, 4))
    x = coverage_df["coverage"]
    y = coverage_df[metric]
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Coverage vs {metric.upper()}")
    ax.set_xlim([0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

