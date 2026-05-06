"""
config.py
---------
Single place for all file paths, column names, and default settings.
The cleaned dataset layer now lives in the top-level datasets/processed folder.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# Input CSVs
TRAIN_CSV = DATA_DIR / "model_construction_dataset.csv"
EXTTEST_CSV = DATA_DIR / "external_test_dataset.csv"

SCAFFOLD_CSVs = {
    "murcko_model": DATA_DIR / "model_molecular_murcko_scaffolds.csv",
    "carbon_model": DATA_DIR / "model_molecular_carbon_scaffolds.csv",
    "murcko_ext": DATA_DIR / "external_molecular_murcko_scaffolds.csv",
    "carbon_ext": DATA_DIR / "external_molecular_carbon_scaffolds.csv",
    "murcko_tox": DATA_DIR / "model_nephrotoxicity_molecular_murcko_scaffolds.csv",
    "carbon_tox": DATA_DIR / "model_nephrotoxicity_molecular_carbon_scaffolds.csv",
}

# ── Column names ──────────────────────────────────────────────────────────────
SMILES_COL = "canonical SMILES"
LABEL_COL = "label"

# ── Featurizer settings ───────────────────────────────────────────────────────
FP_BITS = 1024
PCA_COMPONENTS = 100
BERT_BATCH = 32
BERT_MODEL = "DeepChem/ChemBERTa-77M-MTR"
BERT_FALLBACK = "seyonec/ChemBERTa-zinc-base-v2"

# ── Model settings ────────────────────────────────────────────────────────────
CV_FOLDS = 5
LGBM_TRIALS = 50
GPC_MAX_N = 2000
RANDOM_SEED = 42
THRESHOLD = 0.5
DEVICE = "cpu"

# ── Paper benchmark (Liu et al. 2025, Table 1 & 3) ───────────────────────────
PAPER_INTERNAL = {"auc": 0.933, "acc": 0.852, "se": 0.852, "f1": 0.853, "kappa": 0.703}
PAPER_EXTERNAL = {"auc": 0.846, "acc": 0.848, "se": 0.901, "f1": 0.852, "kappa": 0.697}
PAPER_EXTERNAL_UQ = {"auc": 0.868, "acc": 0.878, "se": 0.940, "f1": 0.877, "kappa": 0.756}
