from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config.settings import LEGACY_NOTEWORTHY_DIR
from src.evaluation.schema import make_summary_row
from src.models.common import ModelResult, combine_notes, locate_existing, prediction_rows_from_frame, threshold_predictions
from src.utils.runners import run_python_script

LEGACY_DIR = LEGACY_NOTEWORTHY_DIR / "nephrotox_modular"
RESULTS_DIR = LEGACY_DIR / "results"


def _summary_rows(model_name: str, metrics_path: Path) -> list[dict[str, object]]:
    # -------------------------------------------------------------------------
    # Modular Family Metric Loader
    # The modular branch stores metrics as JSON blobs across several candidate
    # model families. This helper converts those archived records into the
    # common benchmark schema so early LightGBM, HistGB, GIN, NODE, and
    # stacking experiments remain comparable with the final shortlist.
    # -------------------------------------------------------------------------
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    runtime = metrics.get("runtime_seconds")
    rows: list[dict[str, object]] = []

    cv = metrics.get("cv", {})
    if cv:
        rows.append(
            make_summary_row(
                model_name=model_name,
                variant="cv_mean",
                dataset="internal_cv",
                accuracy=cv.get("acc"),
                recall=cv.get("recall"),
                f1=cv.get("f1"),
                roc_auc=cv.get("auc", cv.get("auroc")),
                pr_auc=cv.get("auprc"),
                training_time=runtime,
                notes="saved_modular_metrics_json",
            )
        )

    ext = metrics.get("ext", {})
    if ext:
        rows.append(
            make_summary_row(
                model_name=model_name,
                variant="all",
                dataset="external",
                accuracy=ext.get("acc"),
                recall=ext.get("recall"),
                f1=ext.get("f1"),
                roc_auc=ext.get("auc", ext.get("auroc")),
                pr_auc=ext.get("auprc"),
                inference_time=runtime,
                notes="detailed_predictions_available",
            )
        )

    ext_hc = metrics.get("ext_hc", {})
    if ext_hc:
        rows.append(
            make_summary_row(
                model_name=model_name,
                variant="high_confidence",
                dataset="external",
                accuracy=ext_hc.get("acc"),
                recall=ext_hc.get("recall"),
                f1=ext_hc.get("f1"),
                roc_auc=ext_hc.get("auc", ext_hc.get("auroc")),
                pr_auc=ext_hc.get("auprc"),
                inference_time=runtime,
                notes="detailed_predictions_unavailable_for_high_confidence_subset",
            )
        )

    return rows


def _prediction_rows(model_name: str, result_dir: Path) -> list[dict[str, object]]:
    # External probabilities are rehydrated from NumPy arrays produced by the
    # original modular runners.
    probs = np.load(result_dir / "ext_proba.npy")
    y_true = np.load(result_dir / "y_ext.npy")
    frame = {
        "y_true": y_true.astype(int),
        "y_pred": threshold_predictions(probs, 0.5),
        "y_prob": probs.astype(float),
    }
    import pandas as pd

    df = pd.DataFrame(frame)
    return prediction_rows_from_frame(
        df,
        model_name=model_name,
        variant="all",
        dataset="external",
        true_col="y_true",
        pred_col="y_pred",
        score_col="y_prob",
        sample_prefix="external",
    )


def _run_modular_script(script_name: str, python_executable: str | None) -> None:
    # Execute the requested modular legacy script in its native working
    # directory so archived imports and relative paths behave as designed.
    run_python_script(
        LEGACY_DIR / script_name,
        cwd=LEGACY_DIR,
        python_executable=python_executable,
    )


def _run_registered(config: dict[str, object], *, script_name: str, result_folder: str, model_name: str) -> ModelResult:
    # Generic wrapper shared across the modular legacy candidates.
    force = bool(config.get("force_rerun", False))
    python_exec = str(config.get("python_executable")) if config.get("python_executable") else None

    metrics_path = RESULTS_DIR / result_folder / "metrics.json"
    if force or not metrics_path.exists():
        _run_modular_script(script_name, python_exec)

    metrics_path = locate_existing(metrics_path)
    result_dir = metrics_path.parent
    summary_rows = _summary_rows(model_name, metrics_path)

    prediction_rows: list[dict[str, object]] = []
    if (result_dir / "ext_proba.npy").exists() and (result_dir / "y_ext.npy").exists():
        prediction_rows = _prediction_rows(model_name, result_dir)
    else:
        summary_rows = [
            {
                **row,
                "notes": combine_notes(row.get("notes"), "detailed_predictions_unavailable"),
            }
            for row in summary_rows
        ]

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def run_modular_lightgbm(config: dict[str, object]) -> ModelResult:
    # Sparse/tabular boosted-tree modular benchmark.
    return _run_registered(config, script_name="run_lightgbm.py", result_folder="LightGBM", model_name="modular_lightgbm")


def run_modular_histgb(config: dict[str, object]) -> ModelResult:
    # Histogram-based gradient boosting variant from the modular phase.
    return _run_registered(
        config,
        script_name="run_histgb.py",
        result_folder="HistGradientBoosting",
        model_name="modular_histgb",
    )


def run_modular_tanimoto_gpc(config: dict[str, object]) -> ModelResult:
    # Similarity-kernel Gaussian Process benchmark from the modular phase.
    return _run_registered(
        config,
        script_name="run_tanimoto_gpc.py",
        result_folder="TanimotoGPC",
        model_name="modular_tanimoto_gpc",
    )


def run_modular_gin_virtual(config: dict[str, object]) -> ModelResult:
    # Graph neural network benchmark with a virtual-node style architecture.
    return _run_registered(
        config,
        script_name="run_gin_virtual.py",
        result_folder="GIN_VirtualNode",
        model_name="modular_gin_virtual",
    )


def run_modular_node(config: dict[str, object]) -> ModelResult:
    # NODE-style differentiable tree ensemble benchmark.
    return _run_registered(config, script_name="run_node.py", result_folder="NODE", model_name="modular_node")


def run_modular_stacking_ensemble(config: dict[str, object]) -> ModelResult:
    # Early multi-model stacking benchmark that prefigured the final ensembles.
    return _run_registered(
        config,
        script_name="run_ensemble.py",
        result_folder="StackingEnsemble",
        model_name="modular_stacking_ensemble",
    )
