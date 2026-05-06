from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.schema import make_prediction_row


# -------------------------------------------------------------------------
# Cross-Family Result Container
# Model runners in this project return both compact summary metrics and the
# per-compound prediction rows needed for downstream comparison, overlap
# inspection, and dashboard search. This lightweight dataclass provides the
# common interface that lets heterogeneous model families plug into one
# orchestration pipeline.
# -------------------------------------------------------------------------
@dataclass
class ModelResult:
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    prediction_rows: list[dict[str, Any]] = field(default_factory=list)


# -------------------------------------------------------------------------
# Archived Artefact Resolution
# The project deliberately tolerates multiple possible result locations
# because the same code may run against fresh outputs or archived benchmark
# exports. This helper implements that reproducibility-oriented fallback.
# -------------------------------------------------------------------------
def locate_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    rendered = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find any expected file: {rendered}")


# -------------------------------------------------------------------------
# Sample Identifier Construction
# External predictions come from many sources: canonical SMILES strings,
# generic row numbers, or legacy CSV identifiers. Stable sample IDs are
# required so predictions can be aligned across engines during external
# validation and in the saved-search dashboard.
# -------------------------------------------------------------------------
def sample_ids_from_frame(
    frame: pd.DataFrame,
    *,
    sample_id_col: str | None = None,
    prefix: str = "sample",
) -> list[str]:
    if sample_id_col and sample_id_col in frame.columns:
        values = frame[sample_id_col].fillna("").astype(str)
        return [value if value else f"{prefix}_{idx + 1:04d}" for idx, value in enumerate(values)]
    return [f"{prefix}_{idx + 1:04d}" for idx in range(len(frame))]


def prediction_rows_from_frame(
    frame: pd.DataFrame,
    *,
    model_name: str,
    variant: str,
    dataset: str,
    true_col: str,
    pred_col: str,
    score_col: str,
    sample_id_col: str | None = None,
    sample_prefix: str = "sample",
) -> list[dict[str, Any]]:
    sample_ids = sample_ids_from_frame(frame, sample_id_col=sample_id_col, prefix=sample_prefix)
    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        rows.append(
            make_prediction_row(
                model_name=model_name,
                variant=variant,
                dataset=dataset,
                sample_id=sample_ids[idx],
                true_label=row.get(true_col),
                predicted_label=row.get(pred_col),
                predicted_score=row.get(score_col),
            )
        )
    return rows


# -------------------------------------------------------------------------
# Binary Classification Metrics
# The project reports the standard medicinal-ML metrics used throughout the
# dissertation: accuracy, precision, recall, F1, ROC-AUC, and PR-AUC. This
# helper recomputes them from prediction rows so archived exports can be
# checked for consistency and supplemented when saved files omit fields.
# -------------------------------------------------------------------------
def compute_binary_metrics(
    *,
    true_labels: Iterable[Any],
    predicted_labels: Iterable[Any],
    predicted_scores: Iterable[Any] | None = None,
) -> dict[str, float | None]:
    y_true = pd.Series(list(true_labels)).dropna().astype(int)
    y_pred = pd.Series(list(predicted_labels)).dropna().astype(int)
    if len(y_true) != len(y_pred) or y_true.empty:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
        }

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
    }

    if predicted_scores is None:
        return metrics

    y_score = pd.Series(list(predicted_scores)).astype(float)
    if len(y_score) != len(y_true):
        return metrics

    if y_true.nunique() > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["pr_auc"] = None

    return metrics


# -------------------------------------------------------------------------
# Metric and Note Merging
# Archived summary rows are often partially populated, especially when they
# originated from different historical experiment branches. These helpers
# preserve recorded metadata while back-filling any computable quantities.
# -------------------------------------------------------------------------
def merge_metrics(row: dict[str, Any], metrics: dict[str, float | None]) -> dict[str, Any]:
    merged = dict(row)
    for key, value in metrics.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
    return merged


def combine_notes(*notes: str | None) -> str | None:
    clean = [note.strip() for note in notes if note and note.strip()]
    if not clean:
        return None
    return "; ".join(clean)


# -------------------------------------------------------------------------
# Probability-to-Class Conversion
# Most archived classifiers export probabilities as the primary output and
# derive a binary call through a threshold. Centralising the thresholding
# step prevents silent inconsistencies across wrappers and live utilities.
# -------------------------------------------------------------------------
def threshold_predictions(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)
