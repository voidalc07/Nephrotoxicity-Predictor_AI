from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)



def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")



def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        return 0.0
    denom = tn + fp
    return float(tn / denom) if denom else 0.0



def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))



def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "auroc": safe_auroc(y_true, y_prob),
        "specificity": specificity_score(y_true, y_pred),
        "sensitivity": sensitivity_score(y_true, y_pred),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
    }



def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_score = -1.0
    best_metrics: Dict[str, float] = {}

    for thr in np.linspace(0.1, 0.9, 81):
        metrics = compute_metrics(y_true, y_prob, float(thr))
        score = 0.60 * metrics["f1"] + 0.40 * metrics["specificity"]
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics
