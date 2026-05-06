from __future__ import annotations

from typing import Any

import pandas as pd

# -------------------------------------------------------------------------
# Normalised Evaluation Schema
# Every model family in KV6013 emits results in its own native format:
# JSON from legacy graph models, CSV from confirmed ensembles, and ad hoc
# arrays from archived noteworthy runs. These canonical column definitions
# impose a common reporting layer so that all families can be compared under
# a single external-validation and dashboard contract.
# -------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "model_name",
    "variant",
    "dataset",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "training_time",
    "inference_time",
    "notes",
]

PREDICTION_COLUMNS = [
    "model_name",
    "variant",
    "dataset",
    "sample_id",
    "true_label",
    "predicted_label",
    "predicted_score",
]


# -------------------------------------------------------------------------
# Metric Type Coercion
# Legacy outputs mix plain floats, strings, nested dictionaries containing
# fold means, and missing values. These helpers defensively coerce values
# into stable numeric types so downstream reporting reflects the benchmark
# semantics rather than file-format accidents.
# -------------------------------------------------------------------------
def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("mean", "value"):
            if key in value:
                return to_float(value[key])
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            return to_float(metrics[key])
    return None


# -------------------------------------------------------------------------
# Summary and Prediction Row Builders
# These constructors create publication-ready rows that can be concatenated
# across model families. Keeping the row shape fixed is essential because
# the dashboard, appendix tables, and saved CSV evidence all consume the
# same merged outputs.
# -------------------------------------------------------------------------
def make_summary_row(
    *,
    model_name: str,
    variant: str | None = None,
    dataset: str = "external",
    accuracy: Any = None,
    precision: Any = None,
    recall: Any = None,
    f1: Any = None,
    roc_auc: Any = None,
    pr_auc: Any = None,
    training_time: Any = None,
    inference_time: Any = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "variant": variant,
        "dataset": dataset,
        "accuracy": to_float(accuracy),
        "precision": to_float(precision),
        "recall": to_float(recall),
        "f1": to_float(f1),
        "roc_auc": to_float(roc_auc),
        "pr_auc": to_float(pr_auc),
        "training_time": to_float(training_time),
        "inference_time": to_float(inference_time),
        "notes": notes,
    }


def make_prediction_row(
    *,
    model_name: str,
    variant: str,
    dataset: str,
    sample_id: Any,
    true_label: Any,
    predicted_label: Any,
    predicted_score: Any,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "variant": variant,
        "dataset": dataset,
        "sample_id": str(sample_id),
        "true_label": to_int(true_label),
        "predicted_label": to_int(predicted_label),
        "predicted_score": to_float(predicted_score),
    }


# -------------------------------------------------------------------------
# Final DataFrame Assembly
# The final pipeline writes a single summary report and a single detailed
# prediction table. These functions guarantee column order and fill absent
# metrics with nulls so that partial legacy runs remain machine-readable.
# -------------------------------------------------------------------------
def finalize_summary_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[SUMMARY_COLUMNS]


def finalize_prediction_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    df = pd.DataFrame(rows)
    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[PREDICTION_COLUMNS]
