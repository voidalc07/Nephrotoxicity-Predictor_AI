from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
# -------------------------------------------------------------------------
# Reproducible Headless Rendering
# The consolidated CLI may trigger plotting code inside archived benchmarking
# scripts. Matplotlib is therefore redirected into a project-local cache so
# figures and font metadata remain reproducible without writing into a user's
# home directory or depending on workstation-specific cache state.
# -------------------------------------------------------------------------
# Keep Matplotlib cache in the project so it does not write to the user's home folder.
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from src.models.registry import ALL_MODELS, MAIN_MODELS, PROTOTYPE_MODELS
from src.pipelines.prepare_datasets import prepare_datasets
from src.pipelines.run_all import parse_models_arg, run_all_models


def build_parser() -> argparse.ArgumentParser:
    # -------------------------------------------------------------------------
    # Unified Experiment Entry Point
    # This parser exposes the portable project as a single executable surface
    # for dataset preparation, model execution, and report aggregation. The
    # intent is architectural rather than algorithmic: the same command can
    # regenerate the dissertation evidence tables, the saved dashboard outputs,
    # and any lightweight reruns required for local verification.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Run the consolidated KV6013 nephrotoxicity models.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of model names. Default: run the main model set.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force model families to rerun instead of reusing archived outputs.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch legacy family pipelines when needed.",
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/final_reports/overall_evaluation.csv",
        help="Path to the combined summary CSV.",
    )
    parser.add_argument(
        "--predictions-csv",
        default="outputs/detailed_predictions/all_model_predictions.csv",
        help="Path to the combined detailed predictions CSV.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the main model names and exit.",
    )
    parser.add_argument(
        "--list-prototype-models",
        action="store_true",
        help="Print the available prototype model names and exit.",
    )
    parser.add_argument(
        "--include-prototypes",
        action="store_true",
        help="Include prototype models in addition to the five main models.",
    )
    parser.add_argument(
        "--prototypes-only",
        action="store_true",
        help="Run only prototype models.",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip rebuilding datasets/processed from datasets/raw before running models.",
    )
    return parser


def main() -> int:
    # -------------------------------------------------------------------------
    # End-to-End Benchmark Orchestration
    # The top-level runner mirrors the methodological flow described in the
    # dissertation: first standardise the source datasets, then select the
    # relevant model family subset, and finally aggregate each model's saved or
    # regenerated outputs into a common evaluation schema suitable for external
    # comparison and dashboard presentation.
    # -------------------------------------------------------------------------
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        for name in MAIN_MODELS.keys():
            print(name)
        return 0

    if args.list_prototype_models:
        for name in PROTOTYPE_MODELS.keys():
            print(name)
        return 0

    if not args.skip_data_prep:
        # ---------------------------------------------------------------------
        # Dataset Curation Before Modelling
        # The project intentionally rebuilds `datasets/processed` before model
        # execution unless told otherwise. This preserves traceability between
        # the raw Liu-style benchmark exports and the canonicalised SMILES /
        # label files consumed by RDKit-driven pipelines downstream.
        # ---------------------------------------------------------------------
        # Clean the data first unless the user asked to skip that step.
        report = prepare_datasets()
        print(
            f"[DATA] Prepared {len(report['datasets'])} datasets "
            f"from {report['raw_dir']} -> {report['processed_dir']}"
        )

    # -------------------------------------------------------------------------
    # Model Family Selection
    # The dissertation distinguishes the five main engines from earlier
    # prototype families. This routing logic preserves that distinction so the
    # user can reproduce either the publication-facing shortlist or the wider
    # historical search space without duplicating orchestration code.
    # -------------------------------------------------------------------------
    # Pick which model group to run before starting the pipeline.
    selected = parse_models_arg(args.models)
    if args.prototypes_only:
        available_models = PROTOTYPE_MODELS
    elif args.include_prototypes or (selected and any(name in PROTOTYPE_MODELS for name in selected)):
        available_models = ALL_MODELS
    else:
        available_models = MAIN_MODELS

    summary_path = Path(args.summary_csv).expanduser()
    predictions_path = Path(args.predictions_csv).expanduser()
    _, _, failures = run_all_models(
        selected_models=selected,
        available_models=available_models,
        force_rerun=args.force_rerun,
        python_executable=args.python_executable,
        summary_csv=summary_path,
        predictions_csv=predictions_path,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
