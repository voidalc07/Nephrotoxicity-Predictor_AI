from __future__ import annotations

import os
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Project Path Resolution and Deployment Profiles
# These constants define the file-system contract of the portable build.
# The codebase supports both a self-contained "portable" mode and a fuller
# research-mode deployment that can reuse archived training artefacts from
# neighbouring repositories. Centralising these paths keeps the scientific
# benchmark assets, live-analysis resources, and dashboard outputs aligned.
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
FULL_BUILD_ROOT = Path(os.environ.get("NEPHROTOX_FULL_BUILD_ROOT", PROJECT_ROOT)).expanduser().resolve()


# -------------------------------------------------------------------------
# Candidate Path Selection
# The portable bundle may be distributed independently of the heavier legacy
# training repository. This helper therefore prefers whichever archive root
# is actually present, allowing the same code to resolve confirmed-model and
# noteworthy-model outputs without hard-coding a single machine-specific path.
# -------------------------------------------------------------------------
def _first_existing(candidates: list[Path], default: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return default

# -------------------------------------------------------------------------
# Dataset, Output, and Archive Layout
# Raw and processed datasets are separated so that the cleaning pipeline can
# be rerun reproducibly. Outputs are likewise separated into final reports,
# detailed predictions, and feedback artefacts, matching the dissertation's
# distinction between preparation, benchmarking, and deployment evidence.
# -------------------------------------------------------------------------
DATASETS_ROOT = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATASETS_ROOT / "raw"
PROCESSED_DATA_DIR = DATASETS_ROOT / "processed"
DATASET_REPORT_PATH = PROCESSED_DATA_DIR / "dataset_cleaning_report.json"

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
FINAL_REPORTS_DIR = OUTPUTS_ROOT / "final_reports"
DETAILED_PREDICTIONS_DIR = OUTPUTS_ROOT / "detailed_predictions"

ARCHIVED_DIR = PROJECT_ROOT / "archived"
_LEGACY_DEFAULT = ARCHIVED_DIR / "legacy_project"
_LEGACY_CANDIDATES = [
    _LEGACY_DEFAULT,
    FULL_BUILD_ROOT / "archived" / "legacy_project",
    PROJECT_ROOT.parent / "FINAL_ KV6013_NEPHROTOXICITY_PREDICTOR" / "archived" / "legacy_project",
    PROJECT_ROOT.parent / "KV6013-Python-Project" / "archived" / "legacy_project",
]
ARCHIVED_LEGACY_DIR = _first_existing(_LEGACY_CANDIDATES, _LEGACY_DEFAULT)
ARCHIVED_REFRACTOR_DIR = ARCHIVED_DIR / "previous_refactor"
RAW_RUNS_DIR = ARCHIVED_DIR / "run_cache"

LEGACY_CONFIRMED_DIR = ARCHIVED_LEGACY_DIR / "confirmed_models"
LEGACY_NOTEWORTHY_DIR = ARCHIVED_LEGACY_DIR / "noteworthy_models"
LEGACY_EXPERIMENTS_DIR = ARCHIVED_LEGACY_DIR / "experiments"
LEGACY_MODELS_DIR = ARCHIVED_LEGACY_DIR / "models"
LEGACY_UTILS_DIR = ARCHIVED_LEGACY_DIR / "utils"

FIXED_DATA_DIR = PROCESSED_DATA_DIR
PROJECT_PLUS_DATA_DIR = PROCESSED_DATA_DIR

DEFAULT_TRAIN_CSV = PROCESSED_DATA_DIR / "model_construction_dataset.csv"
DEFAULT_EXTERNAL_CSV = PROCESSED_DATA_DIR / "external_test_dataset.csv"

PROJECT_PLUS_TRAIN_CSV = DEFAULT_TRAIN_CSV
PROJECT_PLUS_TEST_CSV = DEFAULT_EXTERNAL_CSV

# -------------------------------------------------------------------------
# Live-Inference Assets
# The live screening pathway depends on a lightweight applicability-domain
# context rather than retraining the full research stack at runtime. These
# paths store the precomputed training fingerprint matrix, serialized model
# bundles, and human-in-the-loop feedback CSV used by the live dashboard.
# -------------------------------------------------------------------------
LIVE_ASSETS_DIR = PROJECT_ROOT / "live_assets"
LIVE_ASSET_FINGERPRINTS = LIVE_ASSETS_DIR / "training_fingerprint_context_v1.npz"
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "models"
FEEDBACK_DIR = OUTPUTS_ROOT / "feedback"
FEEDBACK_CSV = FEEDBACK_DIR / "confirmed_live_labels.csv"

# -------------------------------------------------------------------------
# Runtime Environment Flags
# NEPHROTOX_MODE distinguishes portable inference from fuller local builds.
# The active Python executable is also recorded here because several wrapper
# modules spawn legacy scripts whose dependencies may differ from the current
# interpreter environment.
# -------------------------------------------------------------------------
DEPLOYMENT_MODE = os.environ.get("NEPHROTOX_MODE", "portable").strip().lower() or "portable"

PYTHON_EXECUTABLE = sys.executable
