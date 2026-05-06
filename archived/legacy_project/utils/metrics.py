from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Dict, List, Tuple


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = (prob_pos >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    accuracy = metrics.accuracy_score(y_true, preds)
    precision = metrics.precision_score(y_true, preds, zero_division=0)
    recall = metrics.recall_score(y_true, preds, zero_division=0)
    f1 = metrics.f1_score(y_true, preds, zero_division=0)
    auroc = metrics.roc_auc_score(y_true, prob_pos)
    specificity = _safe_div(tn, tn + fp)
    sensitivity = recall
    kappa = metrics.cohen_kappa_score(y_true, preds)
    mcc = metrics.matthews_corrcoef(y_true, preds) if (tp + fp) and (tp + fn) and (tn + fp) and (tn + fn) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "auroc": auroc,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def calibration_scores(y_true: np.ndarray, prob_pos: np.ndarray) -> Dict[str, float]:
    """Return AUROC, Brier score, and log loss for probability estimates."""
    auroc = metrics.roc_auc_score(y_true, prob_pos)
    brier = metrics.brier_score_loss(y_true, prob_pos)
    logloss = metrics.log_loss(y_true, prob_pos, labels=[0, 1])
    return {"auroc": auroc, "brier": brier, "logloss": logloss}


def find_optimal_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    grid: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    best_thresh = grid[0]
    best_metric = -1.0
    best_metrics = {}
    for thr in grid:
        metrics_dict = compute_binary_classification_metrics(y_true, prob_pos, thr)
        score = metrics_dict["f1"]
        if score > best_metric:
            best_metric = score
            best_thresh = thr
            best_metrics = metrics_dict
        elif score == best_metric:
            # Tie-breaker with Youden's J
            youden = metrics_dict["sensitivity"] + metrics_dict["specificity"] - 1
            best_youden = best_metrics.get("sensitivity", 0) + best_metrics.get("specificity", 0) - 1
            if youden > best_youden:
                best_thresh = thr
                best_metrics = metrics_dict
    return best_thresh, best_metrics


def coverage_performance(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    base_threshold: float,
    coverage_thresholds: List[float],
) -> pd.DataFrame:
    rows = []
    max_probs = np.maximum(prob_pos, 1 - prob_pos)
    for cov_thr in coverage_thresholds:
        mask = max_probs >= cov_thr
        coverage = _safe_div(mask.sum(), len(max_probs))
        if mask.sum() == 0:
            row = {"coverage_threshold": cov_thr, "coverage": coverage}
            rows.append(row)
            continue
        filtered_metrics = compute_binary_classification_metrics(y_true[mask], prob_pos[mask], base_threshold)
        # add AUROC on retained subset (guard for single-class)
        try:
            filtered_metrics["auroc"] = metrics.roc_auc_score(y_true[mask], prob_pos[mask])
        except Exception:
            filtered_metrics["auroc"] = np.nan
        filtered_metrics.update({"coverage_threshold": cov_thr, "coverage": coverage})
        rows.append(filtered_metrics)
    return pd.DataFrame(rows)
