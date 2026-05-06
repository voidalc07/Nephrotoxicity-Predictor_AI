from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping

from src.config.settings import DETAILED_PREDICTIONS_DIR, FINAL_REPORTS_DIR, PYTHON_EXECUTABLE, RAW_RUNS_DIR
from src.evaluation.schema import finalize_prediction_results, finalize_summary_results, make_summary_row
from src.models.common import combine_notes
from src.models.registry import ALL_MODELS
from src.utils.io import ensure_dir


# -------------------------------------------------------------------------
# Model Selection Parsing
# The CLI can execute either the five final engines or a wider historical
# prototype set. This parser keeps the interface lightweight while allowing
# reproducible reruns of specific model families from the dissertation.
# -------------------------------------------------------------------------
def parse_models_arg(models_arg: str | None) -> list[str] | None:
    if not models_arg:
        return None
    names = [name.strip() for name in models_arg.split(",") if name.strip()]
    return names or None


# -------------------------------------------------------------------------
# Unified Benchmark Orchestration
# This pipeline is the consolidation layer of the project. It executes the
# requested model families, captures both summary metrics and per-sample
# predictions, and writes harmonised CSV outputs that drive the dashboard,
# appendix tables, and external-validation evidence pack.
# -------------------------------------------------------------------------
def run_all_models(
    *,
    selected_models: list[str] | None = None,
    available_models: Mapping[str, object] | None = None,
    force_rerun: bool = False,
    python_executable: str = PYTHON_EXECUTABLE,
    summary_csv: Path | None = None,
    predictions_csv: Path | None = None,
) -> tuple[Path, Path, int]:
    ensure_dir(RAW_RUNS_DIR)
    ensure_dir(FINAL_REPORTS_DIR)
    ensure_dir(DETAILED_PREDICTIONS_DIR)

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_registry = available_models or ALL_MODELS
    model_names = selected_models or list(model_registry.keys())
    unknown = [name for name in model_names if name not in model_registry]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")

    runtime_config: dict[str, object] = {
        "force_rerun": force_rerun,
        "python_executable": python_executable,
    }

    failures = 0
    for model_name in model_names:
        runner = model_registry[model_name]
        start = time.perf_counter()
        print(f"[RUN] {model_name}")
        try:
            # Each runner encapsulates one scientific model family and returns
            # a common result structure so heterogeneous methods can be merged.
            result = runner(runtime_config)
            elapsed = time.perf_counter() - start

            if not result.summary_rows:
                result.summary_rows = [
                    make_summary_row(
                        model_name=model_name,
                        variant="no_output",
                        dataset="external",
                        notes="runner_completed_without_summary_rows",
                    )
                ]

            for row in result.summary_rows:
                row["notes"] = combine_notes(row.get("notes"), f"pipeline_runtime_sec={elapsed:.2f}")

            summary_rows.extend(result.summary_rows)
            prediction_rows.extend(result.prediction_rows)
            print(
                f"[OK ] {model_name} "
                f"({len(result.summary_rows)} summary rows, {len(result.prediction_rows)} prediction rows)"
            )
        except Exception as exc:
            # Failures are recorded as explicit summary rows so that missing
            # results remain visible during comparative reporting.
            failures += 1
            summary_rows.append(
                make_summary_row(
                    model_name=model_name,
                    variant="runner_error",
                    dataset="error",
                    notes=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"[ERR] {model_name}: {exc}")

    summary_df = finalize_summary_results(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=["model_name", "variant", "dataset"], na_position="last").reset_index(
            drop=True
        )

    predictions_df = finalize_prediction_results(prediction_rows)
    if not predictions_df.empty:
        predictions_df = predictions_df.sort_values(
            by=["model_name", "variant", "dataset", "sample_id"], na_position="last"
        ).reset_index(drop=True)

    summary_target = summary_csv or (FINAL_REPORTS_DIR / "overall_evaluation.csv")
    predictions_target = predictions_csv or (DETAILED_PREDICTIONS_DIR / "all_model_predictions.csv")
    summary_df.to_csv(summary_target, index=False)
    predictions_df.to_csv(predictions_target, index=False)

    print(f"[DONE] Summary report saved: {summary_target}")
    print(f"[DONE] Detailed predictions saved: {predictions_target}")
    print(
        f"[DONE] Summary rows: {len(summary_df)} | Prediction rows: {len(predictions_df)} | Failures: {failures}"
    )
    return summary_target, predictions_target, failures
