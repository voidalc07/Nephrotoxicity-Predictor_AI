from __future__ import annotations

from pathlib import Path

from src.config.settings import DEFAULT_EXTERNAL_CSV, DEFAULT_TRAIN_CSV, LEGACY_NOTEWORTHY_DIR, RAW_RUNS_DIR
from src.models.common import ModelResult, locate_existing
from src.models.legacy.nephro_fixed import _cv_summary_row, _external_rows
from src.utils.runners import run_python_script

LEGACY_DIR = LEGACY_NOTEWORTHY_DIR / "nephrotox_final"
FALLBACK_DIR = LEGACY_NOTEWORTHY_DIR / "nephrotox_fixed"
DEFAULT_OUTPUT_DIR = RAW_RUNS_DIR / "nephrotox_final"


def _run_legacy(output_dir: Path, python_executable: str | None) -> None:
    # -------------------------------------------------------------------------
    # Legacy Noteworthy-Final Launcher
    # This wrapper preserves the original noteworthy-final experiment as a
    # callable component of the portable project instead of rewriting the model
    # logic locally.
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # Archived Result Resolution with Fallback
    # The noteworthy-final branch is partially superseded by the noteworthy-
    # fixed lineage, so the wrapper transparently falls back to the closest
    # equivalent archived outputs when a dedicated final export is absent.
    #
    # NOTE:
    # Falling back to `nephrotox_fixed` preserves availability but means some
    # result rows may reflect an equivalent archived branch rather than a unique
    # standalone noteworthy-final export.
    # -------------------------------------------------------------------------
    force = bool(config.get("force_rerun", False))
    python_exec = str(config.get("python_executable")) if config.get("python_executable") else None

    output_dir = DEFAULT_OUTPUT_DIR
    primary_metrics = output_dir / "external_metrics.csv"
    primary_predictions = output_dir / "external_predictions.csv"
    primary_cv = output_dir / "cv_metrics.csv"
    legacy_metrics = LEGACY_DIR / "results" / "external_metrics.csv"
    legacy_predictions = LEGACY_DIR / "results" / "external_predictions.csv"
    legacy_cv = LEGACY_DIR / "results" / "cv_metrics.csv"
    fallback_metrics = FALLBACK_DIR / "results" / "external_metrics.csv"
    fallback_predictions = FALLBACK_DIR / "results" / "external_predictions.csv"
    fallback_cv = FALLBACK_DIR / "results" / "cv_metrics.csv"

    if force or not (
        (primary_metrics.exists() and primary_predictions.exists())
        or (legacy_metrics.exists() and legacy_predictions.exists())
        or (fallback_metrics.exists() and fallback_predictions.exists())
    ):
        try:
            _run_legacy(output_dir, python_exec)
        except Exception:
            pass

    metrics_path = locate_existing(primary_metrics, legacy_metrics, fallback_metrics)
    predictions_path = locate_existing(primary_predictions, legacy_predictions, fallback_predictions)
    fallback_note = None
    if metrics_path.parent == fallback_metrics.parent:
        fallback_note = "fallback_to_fixed_results_for_equivalent_legacy_code"

    result = _external_rows(
        model_name="noteworthy_final",
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        reused_archived_results=metrics_path.parent in {legacy_metrics.parent, fallback_metrics.parent},
        fallback_note=fallback_note,
    )

    if primary_cv.exists() or legacy_cv.exists() or fallback_cv.exists():
        cv_path = locate_existing(primary_cv, legacy_cv, fallback_cv)
        result.summary_rows.append(_cv_summary_row("noteworthy_final", cv_path))

    return result
