"""
metrics.py
----------
Shared metric computation, per-fold display, summary tables,
paper comparison, and JSON persistence for all model runners.

Every runner imports and uses:
    compute_metrics(y_true, y_pred, y_prob)
    log_fold(fold, n_folds, met, logger)
    log_summary(fold_records, label, logger)
    compare_to_paper(agg, paper_ref, label, logger)
    save_results(model_name, cv_agg, ext_met, ext_hc_met, runtime, oof_proba, ext_proba)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)

import config

logger = logging.getLogger(__name__)

_BLUE   = "#378ADD"
_GREEN  = "#639922"
_AMBER  = "#EF9F27"
_TEAL   = "#1D9E75"
_CORAL  = "#D85A30"
_PURPLE = "#7F77DD"
_GRAY   = "#888780"

METRIC_DISPLAY = {
    "auc":         "AUC (ROC)",
    "acc":         "Accuracy",
    "recall":      "Recall / Sensitivity",
    "specificity": "Specificity",
    "f1":          "F1-score",
    "kappa":       "Cohen's Kappa",
    "mcc":         "MCC",
    "auprc":       "AUPRC",
}


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    """Compute all metrics matching Liu et al. (2025) Table 1 + extras."""
    nc = len(np.unique(y_true))
    auc   = float(roc_auc_score(y_true, y_prob))           if nc > 1 else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if nc > 1 else float("nan")
    acc   = float(accuracy_score(y_true, y_pred))
    rec   = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1    = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc   = float(matthews_corrcoef(y_true, y_pred))
    cm    = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec  = tn / (tn + fp + 1e-9)
    return dict(auc=auc, acc=acc, recall=rec, specificity=spec,
                f1=f1, kappa=kappa, mcc=mcc, auprc=auprc)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_fold(fold: int, n_folds: int, met: dict, logger_) -> None:
    W = 64
    logger_.info(f"    ┌─ Fold {fold}/{n_folds} {'─'*(W-14)}┐")
    logger_.info(f"    │  AUC      : {met['auc']:.4f}    Accuracy    : {met['acc']:.4f}                 │")
    logger_.info(f"    │  Recall   : {met['recall']:.4f}    Specificity : {met['specificity']:.4f}                 │")
    logger_.info(f"    │  F1       : {met['f1']:.4f}    Kappa       : {met['kappa']:.4f}                 │")
    logger_.info(f"    │  MCC      : {met['mcc']:.4f}    AUPRC       : {met['auprc']:.4f}                 │")
    logger_.info(f"    └{'─'*W}┘")


def log_summary(fold_records: List[dict], label: str, logger_) -> Dict[str, dict]:
    """Aggregate fold metrics, log full table, return agg dict."""
    keys = ["auc", "acc", "recall", "specificity", "f1", "kappa", "mcc", "auprc"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in fold_records if not np.isnan(r.get(k, float("nan")))]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    W = 66
    logger_.info("")
    logger_.info("═" * W)
    logger_.info(f"  {label} — Summary ({len(fold_records)} folds)")
    logger_.info("═" * W)
    logger_.info(f"  {'Metric':<24} {'Mean':>8}   {'± Std':>8}")
    logger_.info("  " + "─" * (W - 2))
    for k, disp in METRIC_DISPLAY.items():
        if k in agg:
            logger_.info(f"  {disp:<24} {agg[k]['mean']:>8.4f}   ±{agg[k]['std']:>7.4f}")
    logger_.info("═" * W)
    logger_.info("")
    return agg


def compare_to_paper(agg: dict, paper: dict, label: str, logger_) -> None:
    W = 66
    logger_.info(f"  Head-to-head vs {label}")
    logger_.info("  " + "─" * (W - 2))
    logger_.info(f"  {'Metric':<18} {'Paper':>8}  {'Ours':>8}  {'Delta':>8}  Status")
    logger_.info("  " + "─" * (W - 2))
    label_map = {"auc": "AUC", "acc": "Accuracy", "recall": "Recall",
                 "f1": "F1", "kappa": "Kappa"}
    for k, disp in label_map.items():
        pv = paper.get(k)
        if pv is None:
            continue
        ov = agg.get(k, {}).get("mean", float("nan"))
        d  = ov - pv
        sign = "+" if d >= 0 else ""
        status = "✓ BEAT" if d > 0.005 else ("≈ tied" if d >= -0.005 else "✗ behind")
        logger_.info(f"  {disp:<18} {pv:>8.3f}  {ov:>8.3f}  {sign}{d:>7.3f}  {status}")
    logger_.info("  " + "─" * (W - 2))
    logger_.info("")


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(
    model_name:    str,
    cv_agg:        dict,
    ext_met:       dict,
    ext_hc_met:    dict,
    runtime_sec:   float,
    oof_proba:     Optional[np.ndarray] = None,
    ext_proba:     Optional[np.ndarray] = None,
    y_train:       Optional[np.ndarray] = None,
    y_ext:         Optional[np.ndarray] = None,
) -> Path:
    """Save metrics JSON + optional OOF/ext probability arrays."""
    out_dir = config.RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": model_name,
        "runtime_seconds": round(runtime_sec, 1),
        "cv":  cv_agg,
        "ext": ext_met,
        "ext_hc": ext_hc_met,
    }
    path = out_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {path}")

    if oof_proba is not None:
        np.save(out_dir / "oof_proba.npy", oof_proba.astype(np.float32))
    if ext_proba is not None:
        np.save(out_dir / "ext_proba.npy", ext_proba.astype(np.float32))
    if y_train is not None:
        np.save(out_dir / "y_train.npy", y_train.astype(np.int32))
    if y_ext is not None:
        np.save(out_dir / "y_ext.npy", y_ext.astype(np.int32))
    return path


# ---------------------------------------------------------------------------
# Diagnostic plots saved to model-specific output directory
# ---------------------------------------------------------------------------

def save_plots(
    model_name: str,
    y_true_cv:  np.ndarray,
    y_prob_cv:  np.ndarray,
    y_true_ext: Optional[np.ndarray] = None,
    y_prob_ext: Optional[np.ndarray] = None,
) -> None:
    out_dir = config.RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
    })

    # ROC (CV)
    fpr, tpr, _ = roc_curve(y_true_cv, y_prob_cv)
    auc_cv = roc_auc_score(y_true_cv, y_prob_cv)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=_BLUE, lw=2, label=f"CV pooled AUC = {auc_cv:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.07, color=_BLUE)
    ax.axline((0, 0), slope=1, ls="--", color=_GRAY, lw=1, alpha=0.6, label="Random")
    ax.axhline(config.PAPER_INTERNAL["auc"], ls=":", color=_CORAL, lw=1.2,
               label=f"Paper = {config.PAPER_INTERNAL['auc']:.3f}")
    if y_prob_ext is not None and y_true_ext is not None:
        fpr_e, tpr_e, _ = roc_curve(y_true_ext, y_prob_ext)
        auc_ext = roc_auc_score(y_true_ext, y_prob_ext)
        ax.plot(fpr_e, tpr_e, color=_TEAL, lw=1.5, ls="--",
                label=f"External AUC = {auc_ext:.3f}")
    ax.set(xlabel="1 − Specificity", ylabel="Sensitivity",
           title=f"ROC — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=150)
    plt.close()

    # Confusion matrix (CV)
    y_pred_cv = (y_prob_cv >= 0.5).astype(int)
    cm = confusion_matrix(y_true_cv, y_pred_cv)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                cmap=sns.light_palette(_BLUE, as_cmap=True),
                xticklabels=["Non-toxic (0)", "Nephrotoxic (1)"],
                yticklabels=["Non-toxic (0)", "Nephrotoxic (1)"],
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Confusion — {model_name} (CV pooled)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion.png", dpi=150)
    plt.close()
