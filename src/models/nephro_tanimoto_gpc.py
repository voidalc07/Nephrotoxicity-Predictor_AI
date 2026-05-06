from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import LEGACY_NOTEWORTHY_DIR
from src.evaluation.schema import make_summary_row
from src.models.common import ModelResult, locate_existing, prediction_rows_from_frame, threshold_predictions
from src.utils.runners import run_python_script

LEGACY_DIR = LEGACY_NOTEWORTHY_DIR / "nephrotox_modular"
RESULT_DIR = LEGACY_DIR / "results" / "TanimotoGPC"
MODEL_NAME = "modular_tanimoto_gpc"
MODEL_VARIANT = "TanimotoGPC"


def _load_metrics(metrics_path: Path) -> tuple[list[dict[str, object]], float | None]:
    # -------------------------------------------------------------------------
    # Gaussian-Process Similarity Metrics
    # The legacy Tanimoto model reports internal and external statistics in a
    # compact JSON structure. Those metrics are preserved because the model is
    # methodologically important: it grounds nephrotoxicity prediction in
    # fingerprint similarity and a Gaussian Process classifier rather than pure
    # feature extrapolation.
    # -------------------------------------------------------------------------
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    runtime = metrics.get("runtime_seconds")
    rows: list[dict[str, object]] = []
    cv = metrics.get("cv", {})
    if cv:
        rows.append(
            make_summary_row(
                model_name=MODEL_NAME,
                variant=MODEL_VARIANT,
                dataset="internal_cv",
                accuracy=cv.get("acc"),
                recall=cv.get("recall"),
                f1=cv.get("f1"),
                roc_auc=cv.get("auc", cv.get("auroc")),
                pr_auc=cv.get("auprc"),
                training_time=runtime,
                notes="selected_main_model_from_modular_family",
            )
        )

    ext = metrics.get("ext", {})
    if ext:
        rows.append(
            make_summary_row(
                model_name=MODEL_NAME,
                variant=MODEL_VARIANT,
                dataset="external",
                accuracy=ext.get("acc"),
                recall=ext.get("recall"),
                f1=ext.get("f1"),
                roc_auc=ext.get("auc", ext.get("auroc")),
                pr_auc=ext.get("auprc"),
                inference_time=runtime,
                notes="selected_main_model_from_modular_family; detailed_predictions_available",
            )
        )

    return rows, runtime


def _load_predictions(result_dir: Path) -> list[dict[str, object]]:
    # External probabilities are reconstructed from the saved NumPy arrays so
    # the similarity model can participate in the same per-molecule interface
    # as the ensemble and representation-learning families.
    probs = np.load(result_dir / "ext_proba.npy")
    y_true = np.load(result_dir / "y_ext.npy")
    frame = pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "y_pred": threshold_predictions(probs, 0.5),
            "y_prob": probs.astype(float),
        }
    )
    return prediction_rows_from_frame(
        frame,
        model_name=MODEL_NAME,
        variant=MODEL_VARIANT,
        dataset="external",
        true_col="y_true",
        pred_col="y_pred",
        score_col="y_prob",
        sample_prefix="external",
    )


def run_registered(config: dict[str, object]) -> ModelResult:
    # -------------------------------------------------------------------------
    # Archived Tanimoto GPC Reuse
    # This wrapper keeps the original similarity-kernel implementation intact.
    # The live portable runtime later uses a cheaper similarity fallback for
    # responsiveness, but the benchmark table here still reflects the true
    # Gaussian Process evaluation from the legacy project.
    # -------------------------------------------------------------------------
    force = bool(config.get("force_rerun", False))
    python_exec = str(config.get("python_executable")) if config.get("python_executable") else None

    metrics_path = RESULT_DIR / "metrics.json"
    if force or not metrics_path.exists():
        run_python_script(
            LEGACY_DIR / "run_tanimoto_gpc.py",
            cwd=LEGACY_DIR,
            python_executable=python_exec,
        )

    metrics_path = locate_existing(metrics_path)
    summary_rows, _ = _load_metrics(metrics_path)

    prediction_rows: list[dict[str, object]] = []
    if (RESULT_DIR / "ext_proba.npy").exists() and (RESULT_DIR / "y_ext.npy").exists():
        prediction_rows = _load_predictions(RESULT_DIR)

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)
