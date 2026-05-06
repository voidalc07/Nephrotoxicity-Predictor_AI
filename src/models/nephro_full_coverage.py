from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.settings import LEGACY_CONFIRMED_DIR, PROJECT_PLUS_TEST_CSV, PROJECT_PLUS_TRAIN_CSV, RAW_RUNS_DIR
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

IMPROVEMENTS_MAIN_VARIANT = "meta_logreg_knn_plus_lgbm__cal_none__thr_f1"
DESCRIPTOR_MAIN_VARIANT = "challenger_proxy_meta_logreg_knn_plus_lgbm__cal_none__thr_f1"


def _prediction_lookup(root: Path, prefix: str) -> dict[str, Path]:
    # -------------------------------------------------------------------------
    # Variant-to-File Resolution
    # The archived ensemble experiments emit one CSV per candidate variant. This
    # helper reconstructs that mapping so the portable wrapper can preserve the
    # original model-selection story rather than exposing only a single final
    # checkpoint.
    # -------------------------------------------------------------------------
    lookup: dict[str, Path] = {}
    for path in root.rglob("*.csv"):
        if path.name.startswith(prefix):
            lookup[path.stem[len(prefix) :]] = path
    return lookup


def _external_prediction_rows(
    *,
    model_name: str,
    variant: str,
    predictions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    # Convert archived external prediction files into the shared schema used by
    # the consolidated benchmark tables and dashboard search results.
    frame = safe_read_csv(predictions_path)
    pred_col = "pred_label" if "pred_label" in frame.columns else "pred"
    score_col = "prob_nephrotoxic" if "prob_nephrotoxic" in frame.columns else "prob"
    rows = prediction_rows_from_frame(
        frame,
        model_name=model_name,
        variant=variant,
        dataset="external",
        true_col="label",
        pred_col=pred_col,
        score_col=score_col,
        sample_id_col="canonical_smiles",
        sample_prefix="external",
    )
    metrics = compute_binary_metrics(
        true_labels=frame["label"],
        predicted_labels=frame[pred_col],
        predicted_scores=frame[score_col],
    )
    return rows, metrics


def _collect_improvements(root: Path, *, reused_archived_results: bool) -> ModelResult:
    # -------------------------------------------------------------------------
    # Full-Coverage Ensemble Harvesting
    # The "improvements" family corresponds to the strongest full-coverage
    # stacked systems. These models combine KNN locality, LightGBM non-linear
    # partitioning, calibration choices, and threshold tuning across a broad
    # chemistry feature space intended to maximise external robustness.
    # -------------------------------------------------------------------------
    comparison_path = root / "comparison" / "baseline_vs_all_full_coverage_variants.csv"
    comparison_df = safe_read_csv(comparison_path)
    predictions_lookup = _prediction_lookup(root, "external_predictions__")

    baseline_config = root / "baseline" / "config.json"
    if baseline_config.exists():
        baseline_variant = safe_read_json(baseline_config).get("model_name")
        baseline_predictions = root / "baseline" / "predictions_external.csv"
        if baseline_variant and baseline_predictions.exists():
            predictions_lookup[str(baseline_variant)] = baseline_predictions

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for _, raw_row in comparison_df.iterrows():
        row = raw_row.to_dict()
        variant = str(row["variant"])
        prediction_path = predictions_lookup.get(variant)
        has_details = prediction_path is not None and prediction_path.exists()
        metrics: dict[str, float | None] = {}

        if has_details and prediction_path is not None:
            variant_prediction_rows, metrics = _external_prediction_rows(
                model_name="full_coverage_improvements",
                variant=variant,
                predictions_path=prediction_path,
            )
            prediction_rows.extend(variant_prediction_rows)

        external_row = make_summary_row(
            model_name="full_coverage_improvements",
            variant=variant,
            dataset="external",
            accuracy=row.get("external_accuracy"),
            f1=row.get("external_f1"),
            roc_auc=row.get("external_auroc"),
            notes=combine_notes(
                f"base_model={row.get('base_model')}",
                f"coverage={row.get('coverage')}",
                "detailed_predictions_available" if has_details else "detailed_predictions_unavailable",
                "reused_archived_results" if reused_archived_results else None,
            ),
        )
        summary_rows.append(merge_metrics(external_row, metrics))

        summary_rows.append(
            make_summary_row(
                model_name="full_coverage_improvements",
                variant=variant,
                dataset="internal_cv",
                accuracy=row.get("internal_accuracy_mean"),
                f1=row.get("internal_f1_mean"),
                roc_auc=row.get("internal_auroc_mean"),
                notes=combine_notes(
                    f"base_model={row.get('base_model')}",
                    f"coverage={row.get('coverage')}",
                    "internal_cv_mean",
                ),
            )
        )

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def _collect_descriptor_v4(root: Path, *, reused_archived_results: bool) -> ModelResult:
    # -------------------------------------------------------------------------
    # Descriptor + Fingerprint Ensemble Harvesting
    # This family emphasises curated physicochemical descriptors alongside
    # Morgan fingerprints. It represents the project's strongest chemistry-
    # engineered alternative to the broader full-coverage stack, and is useful
    # scientifically because descriptor subsets can remain interpretable while
    # fingerprints preserve substructural specificity.
    # -------------------------------------------------------------------------
    comparison_path = root / "comparison" / "baseline_vs_all_full_coverage_variants.csv"
    comparison_df = safe_read_csv(comparison_path)
    predictions_lookup = _prediction_lookup(root / "predictions", "predictions_external__")

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for _, raw_row in comparison_df.iterrows():
        row = raw_row.to_dict()
        variant = str(row["variant"])
        prediction_path = predictions_lookup.get(variant)
        has_details = prediction_path is not None and prediction_path.exists()
        metrics: dict[str, float | None] = {}

        if has_details and prediction_path is not None:
            variant_prediction_rows, metrics = _external_prediction_rows(
                model_name="full_coverage_descriptor_fp_meta_v4",
                variant=variant,
                predictions_path=prediction_path,
            )
            prediction_rows.extend(variant_prediction_rows)

        external_row = make_summary_row(
            model_name="full_coverage_descriptor_fp_meta_v4",
            variant=variant,
            dataset="external",
            accuracy=row.get("external_accuracy"),
            recall=row.get("external_recall"),
            f1=row.get("external_f1"),
            roc_auc=row.get("external_auroc"),
            notes=combine_notes(
                f"family={row.get('family')}",
                f"selection_mode={row.get('selection_mode')}",
                f"coverage={row.get('coverage')}",
                "detailed_predictions_available" if has_details else "detailed_predictions_unavailable",
                "reused_archived_results" if reused_archived_results else None,
            ),
        )
        summary_rows.append(merge_metrics(external_row, metrics))

        summary_rows.append(
            make_summary_row(
                model_name="full_coverage_descriptor_fp_meta_v4",
                variant=variant,
                dataset="internal_cv",
                accuracy=row.get("internal_accuracy_mean"),
                recall=row.get("internal_recall_mean"),
                f1=row.get("internal_f1_mean"),
                roc_auc=row.get("internal_auroc_mean"),
                notes=combine_notes(
                    f"family={row.get('family')}",
                    f"selection_mode={row.get('selection_mode')}",
                    "internal_cv_mean",
                ),
            )
        )

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def _select_main_variant(result: ModelResult, *, selected_variant: str) -> ModelResult:
    # Mark the dissertation-shortlisted ensemble variant without discarding the
    # archived comparison results from which it was chosen.
    summary_rows = [row for row in result.summary_rows if row.get("variant") == selected_variant]
    prediction_rows = [row for row in result.prediction_rows if row.get("variant") == selected_variant]

    if not summary_rows:
        raise ValueError(f"Could not find selected variant {selected_variant!r} in full-coverage results.")

    for row in summary_rows:
        row["notes"] = combine_notes(row.get("notes"), "selected_main_variant")

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def run_full_coverage_improvements_all_variants(config: dict[str, object]) -> ModelResult:
    # -------------------------------------------------------------------------
    # Portable Wrapper Around the Legacy Training Script
    # The portable project does not re-implement the full legacy modelling
    # stack here; instead it reruns or reuses the original archived pipeline so
    # reported scores remain aligned with the research code that produced them.
    # -------------------------------------------------------------------------
    force = bool(config.get("force_rerun", False))
    python_exec = config.get("python_executable")

    output_root = RAW_RUNS_DIR / "full_coverage_improvements"
    primary = output_root / "comparison" / "baseline_vs_all_full_coverage_variants.csv"
    fallback_root = LEGACY_CONFIRMED_DIR / "results_full_coverage_improvements"
    fallback = fallback_root / "comparison" / "baseline_vs_all_full_coverage_variants.csv"

    if force or not (primary.exists() or fallback.exists()):
        run_python_script(
            LEGACY_CONFIRMED_DIR / "run_full_coverage_improvements.py",
            [
                "--train",
                str(PROJECT_PLUS_TRAIN_CSV),
                "--external",
                str(PROJECT_PLUS_TEST_CSV),
                "--output-root",
                str(output_root),
            ],
            cwd=LEGACY_CONFIRMED_DIR.parent,
            python_executable=str(python_exec) if python_exec else None,
        )

    comparison_path = locate_existing(primary, fallback)
    root = comparison_path.parent.parent
    return _collect_improvements(root, reused_archived_results=root == fallback_root)


def run_full_coverage_improvements(config: dict[str, object]) -> ModelResult:
    # Return only the selected dissertation-facing full-coverage engine.
    return _select_main_variant(
        run_full_coverage_improvements_all_variants(config),
        selected_variant=IMPROVEMENTS_MAIN_VARIANT,
    )


def run_full_coverage_descriptor_fp_meta_v4_all_variants(config: dict[str, object]) -> ModelResult:
    # The descriptor-focused family is wrapped separately because its archived
    # search space and selected variant differ from the broader improvements
    # stack even though both are ensemble-based.
    force = bool(config.get("force_rerun", False))
    python_exec = config.get("python_executable")

    output_root = RAW_RUNS_DIR / "full_coverage_descriptor_fp_meta_v4"
    primary = output_root / "comparison" / "baseline_vs_all_full_coverage_variants.csv"
    fallback_root = LEGACY_CONFIRMED_DIR / "results_full_coverage_descriptor_fp_meta_v4"
    fallback = fallback_root / "comparison" / "baseline_vs_all_full_coverage_variants.csv"

    if force or not (primary.exists() or fallback.exists()):
        run_python_script(
            LEGACY_CONFIRMED_DIR / "run_full_coverage_descriptor_fp_meta_v4.py",
            [
                "--train",
                str(PROJECT_PLUS_TRAIN_CSV),
                "--external",
                str(PROJECT_PLUS_TEST_CSV),
                "--output-root",
                str(output_root),
            ],
            cwd=LEGACY_CONFIRMED_DIR.parent,
            python_executable=str(python_exec) if python_exec else None,
        )

    comparison_path = locate_existing(primary, fallback)
    root = comparison_path.parent.parent
    return _collect_descriptor_v4(root, reused_archived_results=root == fallback_root)


def run_full_coverage_descriptor_fp_meta_v4(config: dict[str, object]) -> ModelResult:
    # Return only the shortlisted descriptor + fingerprint engine used on the
    # main dashboard and in the final comparison tables.
    return _select_main_variant(
        run_full_coverage_descriptor_fp_meta_v4_all_variants(config),
        selected_variant=DESCRIPTOR_MAIN_VARIANT,
    )
