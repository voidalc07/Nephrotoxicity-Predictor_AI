from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import DEFAULT_EXTERNAL_CSV, DEFAULT_TRAIN_CSV, LEGACY_NOTEWORTHY_DIR, RAW_RUNS_DIR
from src.evaluation.schema import make_summary_row
from src.models.common import (
    ModelResult,
    combine_notes,
    compute_binary_metrics,
    locate_existing,
    merge_metrics,
    prediction_rows_from_frame,
)
from src.utils.io import safe_read_csv
from src.utils.runners import run_python_script

LEGACY_DIR = LEGACY_NOTEWORTHY_DIR / "nephrotox_fixed"
DEFAULT_OUTPUT_DIR = RAW_RUNS_DIR / "nephrotox_fixed"


def _cv_summary_row(model_name: str, cv_path: Path) -> dict[str, object]:
    # Summarise saved cross-validation folds into a single internal benchmark
    # row for the consolidated analytics tables.
    cv_df = safe_read_csv(cv_path)
    return make_summary_row(
        model_name=model_name,
        variant="cv_mean",
        dataset="internal_cv",
        accuracy=cv_df["acc"].mean(),
        recall=cv_df["se"].mean(),
        f1=cv_df["f1"].mean(),
        roc_auc=cv_df["auc"].mean(),
        notes="mean_of_saved_cv_folds",
    )


def _external_rows(
    *,
    model_name: str,
    metrics_path: Path,
    predictions_path: Path,
    reused_archived_results: bool,
    fallback_note: str | None = None,
) -> ModelResult:
    # -------------------------------------------------------------------------
    # Noteworthy-Fixed Result Packaging
    # The noteworthy-fixed branch exports both "all" predictions and a
    # high-confidence subset. Both are retained because they reflect an earlier
    # calibration / selective-prediction perspective explored in the project.
    # -------------------------------------------------------------------------
    metrics_df = safe_read_csv(metrics_path)
    predictions_df = safe_read_csv(predictions_path)

    prediction_rows = prediction_rows_from_frame(
        predictions_df,
        model_name=model_name,
        variant="all",
        dataset="external",
        true_col="y_true",
        pred_col="y_pred",
        score_col="prob_nephrotoxic",
        sample_prefix="external",
    )

    high_conf_df = predictions_df[predictions_df["high_confidence"] == 1].reset_index(drop=True)
    if not high_conf_df.empty:
        prediction_rows.extend(
            prediction_rows_from_frame(
                high_conf_df,
                model_name=model_name,
                variant="high_confidence",
                dataset="external",
                true_col="y_true",
                pred_col="y_pred",
                score_col="prob_nephrotoxic",
                sample_prefix="external_hc",
            )
        )

    summary_rows: list[dict[str, object]] = []
    variant_frames = {
        "all": predictions_df,
        "high_confidence": high_conf_df,
    }
    for _, raw_row in metrics_df.iterrows():
        row = raw_row.to_dict()
        variant = str(row.get("split", "all"))
        variant_frame = variant_frames.get(variant, pd.DataFrame())
        metrics = compute_binary_metrics(
            true_labels=variant_frame["y_true"] if not variant_frame.empty else [],
            predicted_labels=variant_frame["y_pred"] if not variant_frame.empty else [],
            predicted_scores=variant_frame["prob_nephrotoxic"] if not variant_frame.empty else [],
        )
        summary_row = make_summary_row(
            model_name=model_name,
            variant=variant,
            dataset="external",
            accuracy=row.get("acc"),
            recall=row.get("se"),
            f1=row.get("f1"),
            roc_auc=row.get("auc"),
            pr_auc=row.get("auprc"),
            notes=combine_notes(
                f"specificity={row.get('specificity')}",
                f"mcc={row.get('mcc')}",
                "detailed_predictions_available",
                fallback_note,
                "reused_archived_results" if reused_archived_results else None,
            ),
        )
        summary_rows.append(merge_metrics(summary_row, metrics))

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def _run_legacy(output_dir: Path, python_executable: str | None) -> None:
    # Launch the original noteworthy-fixed training/evaluation script using the
    # consolidated processed train and external benchmark CSVs.
    run_python_script(
        LEGACY_DIR / "main.py",
        [
            "--mode",
            "train_eval",
            "--train_csv",
            str(DEFAULT_TRAIN_CSV),
            "--ext_csv",
            str(DEFAULT_EXTERNAL_CSV),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=LEGACY_DIR,
        python_executable=python_executable,
    )


def run_registered(config: dict[str, object]) -> ModelResult:
    # Rerun when needed, otherwise reuse the archived noteworthy-fixed outputs.
    force = bool(config.get("force_rerun", False))
    python_exec = str(config.get("python_executable")) if config.get("python_executable") else None

    output_dir = DEFAULT_OUTPUT_DIR
    primary_metrics = output_dir / "external_metrics.csv"
    primary_predictions = output_dir / "external_predictions.csv"
    primary_cv = output_dir / "cv_metrics.csv"
    fallback_metrics = LEGACY_DIR / "results" / "external_metrics.csv"
    fallback_predictions = LEGACY_DIR / "results" / "external_predictions.csv"
    fallback_cv = LEGACY_DIR / "results" / "cv_metrics.csv"

    if force or not (
        (primary_metrics.exists() and primary_predictions.exists())
        or (fallback_metrics.exists() and fallback_predictions.exists())
    ):
        _run_legacy(output_dir, python_exec)

    metrics_path = locate_existing(primary_metrics, fallback_metrics)
    predictions_path = locate_existing(primary_predictions, fallback_predictions)
    result = _external_rows(
        model_name="noteworthy_fixed",
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        reused_archived_results=metrics_path.parent == fallback_metrics.parent,
    )

    if primary_cv.exists() or fallback_cv.exists():
        cv_path = locate_existing(primary_cv, fallback_cv)
        result.summary_rows.append(_cv_summary_row("noteworthy_fixed", cv_path))

    return result
