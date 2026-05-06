from __future__ import annotations

from pathlib import Path

from src.config.settings import LEGACY_NOTEWORTHY_DIR, PROJECT_PLUS_TEST_CSV, PROJECT_PLUS_TRAIN_CSV, RAW_RUNS_DIR
from src.evaluation.schema import make_summary_row
from src.models.common import (
    ModelResult,
    combine_notes,
    compute_binary_metrics,
    locate_existing,
    merge_metrics,
    prediction_rows_from_frame,
)
from src.utils.io import safe_read_csv, safe_read_json
from src.utils.runners import run_python_script

LEGACY_DIR = LEGACY_NOTEWORTHY_DIR / "nephrotox_project_plus"
DEFAULT_OUTPUT_ROOT = RAW_RUNS_DIR / "project_plus"


def _collect(model_name: str, metrics_path: Path, predictions_path: Path, *, reused_archived_results: bool) -> ModelResult:
    # -------------------------------------------------------------------------
    # Project-Plus Result Packaging
    # The project-plus branch explored hybrid graph, sequence, and tabular
    # pipelines. These metrics are preserved because they document a major
    # exploratory phase in which transformer and graph representations were
    # benchmarked against chemistry-engineered baselines.
    # -------------------------------------------------------------------------
    metrics = safe_read_json(metrics_path)
    predictions_df = safe_read_csv(predictions_path)

    prediction_rows = prediction_rows_from_frame(
        predictions_df,
        model_name=f"project_plus_{model_name}",
        variant=model_name,
        dataset="external",
        true_col="y_true",
        pred_col="y_pred",
        score_col="y_prob",
        sample_prefix="external",
    )
    external_metrics = compute_binary_metrics(
        true_labels=predictions_df["y_true"],
        predicted_labels=predictions_df["y_pred"],
        predicted_scores=predictions_df["y_prob"],
    )

    summary_rows = [
        make_summary_row(
            model_name=f"project_plus_{model_name}",
            variant=model_name,
            dataset="internal_validation",
            accuracy=metrics.get("val", {}).get("accuracy"),
            precision=metrics.get("val", {}).get("precision"),
            recall=metrics.get("val", {}).get("recall"),
            f1=metrics.get("val", {}).get("f1"),
            roc_auc=metrics.get("val", {}).get("auroc"),
            notes=combine_notes(
                f"threshold={metrics.get('val', {}).get('threshold')}",
                "validation_split_metrics_from_saved_json",
            ),
        ),
        merge_metrics(
            make_summary_row(
                model_name=f"project_plus_{model_name}",
                variant=model_name,
                dataset="external",
                accuracy=metrics.get("test", {}).get("accuracy"),
                precision=metrics.get("test", {}).get("precision"),
                recall=metrics.get("test", {}).get("recall"),
                f1=metrics.get("test", {}).get("f1"),
                roc_auc=metrics.get("test", {}).get("auroc"),
                notes=combine_notes(
                    f"threshold={metrics.get('test', {}).get('threshold')}",
                    "detailed_predictions_available",
                    "reused_archived_results" if reused_archived_results else None,
                ),
            ),
            external_metrics,
        ),
    ]
    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def _run_model(config: dict[str, object], model_name: str) -> ModelResult:
    # Rerun or reuse one project-plus candidate, keeping the original CLI entry
    # point and output structure intact.
    force = bool(config.get("force_rerun", False))
    python_exec = str(config.get("python_executable")) if config.get("python_executable") else None

    output_dir = DEFAULT_OUTPUT_ROOT / model_name
    primary_metrics = output_dir / "metrics.json"
    primary_predictions = output_dir / "predictions.csv"
    fallback_dir = LEGACY_DIR / "outputs_compare" / model_name
    fallback_metrics = fallback_dir / "metrics.json"
    fallback_predictions = fallback_dir / "predictions.csv"

    if force or not (
        (primary_metrics.exists() and primary_predictions.exists())
        or (fallback_metrics.exists() and fallback_predictions.exists())
    ):
        run_python_script(
            LEGACY_DIR / "main.py",
            [
                "--train_csv",
                str(PROJECT_PLUS_TRAIN_CSV),
                "--test_csv",
                str(PROJECT_PLUS_TEST_CSV),
                "--model_name",
                model_name,
                "--output_dir",
                str(output_dir),
                "--prefer_cpu",
            ],
            cwd=LEGACY_DIR,
            python_executable=python_exec,
        )

    metrics_path = locate_existing(primary_metrics, fallback_metrics)
    predictions_path = locate_existing(primary_predictions, fallback_predictions)
    return _collect(
        model_name,
        metrics_path,
        predictions_path,
        reused_archived_results=metrics_path.parent == fallback_dir,
    )


def run_project_plus_gin(config: dict[str, object]) -> ModelResult:
    # Graph-only project-plus baseline.
    return _run_model(config, "gin")


def run_project_plus_gin_hybrid(config: dict[str, object]) -> ModelResult:
    # Hybrid graph/tabular project-plus baseline.
    return _run_model(config, "gin_hybrid")


def run_project_plus_chemberta(config: dict[str, object]) -> ModelResult:
    # Transformer-driven project-plus baseline.
    return _run_model(config, "chemberta")
