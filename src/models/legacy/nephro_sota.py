import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import DEFAULT_EXTERNAL_CSV, DEFAULT_TRAIN_CSV, FIXED_DATA_DIR, LEGACY_CONFIRMED_DIR, RAW_RUNS_DIR
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

DEFAULT_ABS_DATA_DIR = FIXED_DATA_DIR
DEFAULT_REPO_DATA_DIR = FIXED_DATA_DIR
DEFAULT_OUTPUT_DIR = RAW_RUNS_DIR / "nephro_sota"

ALERTS = {
    "Methyl phosphate": "[#6]OP(=O)(O[#6])O[#6]",
    "Glycoside": "[#6]1-O-[#6](-O-[#6])-[#6](-O)-[#6](-O)-[#6]-1-O",
    "Fluoroquinolone": "n1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)cc12",
    "Beta-lactam": "N1C(=O)CC1",
    "Cephalosporin": "O=C1N2C(=C(CS[C@H]2[C@H]1)C)C(=O)O",
    "Tetrazole": "[c,C]1=NN=NN1",
    "Phenylsulfonylacetic acid": "c1ccccc1S(=O)(=O)CC(=O)O",
    "Pyridinecarboxamide": "c1ccncc1C(=O)N",
    "Purine": "n1cnc2c1ncnc2",
    "Chlorobenzene": "c1ccccc1Cl",
}


def resolve_data_dir(explicit: str | None) -> Path:
    # -------------------------------------------------------------------------
    # Data Root Resolution
    # The SOTA-inspired baseline shares the same processed benchmark CSVs as the
    # rest of the portable project so comparisons remain like-for-like.
    # -------------------------------------------------------------------------
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {path}")
        return path
    if DEFAULT_ABS_DATA_DIR.exists():
        return DEFAULT_ABS_DATA_DIR
    if DEFAULT_REPO_DATA_DIR.exists():
        return DEFAULT_REPO_DATA_DIR
    raise FileNotFoundError("Could not find a data directory. Pass --data-dir explicitly.")


def pick_smiles_column(df: pd.DataFrame) -> str:
    # Support minor schema drift across archived CSV exports.
    normalized = {c.strip().lower().replace("_", " "): c for c in df.columns}
    for key in ("canonical smiles", "canonical_smiles", "smiles", "smiles clean"):
        if key in normalized:
            return normalized[key]
    raise ValueError("No SMILES-like column found in dataset.")


def get_scaffold(smiles: str) -> str:
    # -------------------------------------------------------------------------
    # Bemis-Murcko Scaffold Extraction
    # Murcko scaffolds are used to define chemistry-aware grouping for
    # cross-validation. This reduces optimistic leakage caused by close analogs
    # appearing in both training and validation folds.
    # -------------------------------------------------------------------------
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def get_features(smiles: str) -> dict[str, float] | None:
    # Descriptor, Morgan fingerprint, and SMARTS-alert fusion creates a rich
    # tabular chemistry representation for the CatBoost optimiser to explore.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = {d[0]: d[1](mol) for d in Descriptors._descList}

    from rdkit.Chem import rdFingerprintGenerator

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = list(mfpgen.GetFingerprint(mol))
    for i, bit in enumerate(fp):
        desc[f"fp_{i}"] = int(bit)

    for name, smarts in ALERTS.items():
        patt = Chem.MolFromSmarts(smarts)
        desc[name] = 1 if (patt and mol.HasSubstructMatch(patt)) else 0

    return desc


def load_and_preprocess(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    # Convert canonicalised SMILES into model-ready features while keeping the
    # original strings for scaffold analysis and external prediction export.
    df = pd.read_csv(path)
    smiles_col = pick_smiles_column(df)
    print(f"Loading {path}, shape: {df.shape}")

    features, labels, valid_smiles = [], [], []
    for _, row in df.iterrows():
        f = get_features(str(row[smiles_col]))
        if f is not None:
            features.append(f)
            labels.append(int(row["label"]))
            valid_smiles.append(str(row[smiles_col]))

    return pd.DataFrame(features), np.array(labels), valid_smiles


def main() -> None:
    # -------------------------------------------------------------------------
    # SOTA-Inspired CatBoost Baseline with Optuna
    # This branch tests whether a modern boosted-tree learner over descriptors,
    # Morgan fingerprints, and SMARTS alerts can approach or exceed the final
    # ensemble performance. GroupKFold over Bemis-Murcko scaffolds is used
    # during tuning to make the validation procedure chemically stricter.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="SOTA descriptor+FP+alerts CatBoost baseline.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    train_path = Path(args.train_path) if args.train_path else data_dir / "model construction dataset.csv"
    test_path = Path(args.test_path) if args.test_path else data_dir / "external test dataset.csv"

    print("Extracting features for training set...")
    X_train, y_train, train_smiles = load_and_preprocess(train_path)
    print("Extracting features for external test set...")
    X_test, y_test, test_smiles = load_and_preprocess(test_path)

    print("Checking for leakage...")
    # Exact SMILES overlap is removed before fitting so the external evaluation
    # remains a genuine holdout rather than a memorisation test.
    overlap = set(train_smiles).intersection(set(test_smiles))
    if overlap:
        print(f"Found {len(overlap)} overlapping molecules. Removing them from training set.")
        mask = np.array([s not in overlap for s in train_smiles], dtype=bool)
        X_train = X_train[mask].reset_index(drop=True)
        y_train = y_train[mask]
        train_smiles = [s for s in train_smiles if s not in overlap]

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print("Calculating scaffolds for training set...")
    train_scaffolds = [get_scaffold(s) for s in train_smiles]

    # ---------------------------------------------------------------------
    # Optuna Search Objective
    # Hyperparameters are tuned against scaffold-grouped validation AUROC so
    # optimisation rewards transfer across scaffold families rather than merely
    # analog-level interpolation.
    # ---------------------------------------------------------------------
    def objective(trial: optuna.Trial) -> float:
        param = {
            "objective": "Logloss",
            "eval_metric": "AUC",
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.2),
            "depth": trial.suggest_int("depth", 4, 10),
            "boosting_type": "Plain",
            "bootstrap_type": "MVS",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "iterations": 500,
            "task_type": "CPU",
            "thread_count": -1,
            "verbose": False,
            "early_stopping_rounds": 20,
            "random_seed": args.seed,
        }

        gbm = CatBoostClassifier(**param)
        gkf = GroupKFold(n_splits=5)
        aucs = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=train_scaffolds):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
            gbm.fit(X_t, y_t, eval_set=(X_v, y_v))
            preds = gbm.predict_proba(X_v)[:, 1]
            aucs.append(roc_auc_score(y_v, preds))
        return float(np.mean(aucs))

    print(f"Starting Optuna optimization ({args.n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    print(f"Best cross-validation AUROC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = dict(study.best_params)
    best_params.update(
        {
            "objective": "Logloss",
            "eval_metric": "AUC",
            "iterations": args.iterations,
            "task_type": "CPU",
            "thread_count": -1,
            "verbose": False,
            "random_seed": args.seed,
        }
    )

    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train)

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_binary = model.predict(X_test).astype(int)

    metrics = {
        "model_name": "nephro_sota_catboost",
        "auroc": float(roc_auc_score(y_test, preds_proba)),
        "accuracy": float(accuracy_score(y_test, preds_binary)),
        "kappa": float(cohen_kappa_score(y_test, preds_binary)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "overlap_removed": int(len(overlap)),
        "optuna_best_cv_auroc": float(study.best_value),
    }

    print("\n--- Final Results on External Set ---")
    print(f"AUROC:    {metrics['auroc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Kappa:    {metrics['kappa']:.4f}")
    # NOTE:
    # The success/fail message reflects an experiment-specific target rather
    # than a neutral reporting convention. It is preserved for fidelity to the
    # original exploratory script.
    print("\nSUCCESS: Target AUROC 0.855 exceeded!" if metrics["auroc"] > 0.855 else "\nFAILED: Did not exceed target AUROC 0.855.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(out_dir / "external_metrics.csv", index=False)
    pd.DataFrame(
        {
            "canonical_smiles": test_smiles,
            "label": y_test,
            "prob": preds_proba,
            "pred": preds_binary,
        }
    ).to_csv(out_dir / "external_predictions.csv", index=False)
    pd.DataFrame(study.trials_dataframe()).to_csv(out_dir / "optuna_trials.csv", index=False)
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(f"Saved outputs to: {out_dir.resolve()}")


def _collect_registered_result(
    metrics_path: Path,
    predictions_path: Path,
    *,
    reused_archived_results: bool,
) -> ModelResult:
    # Repackage the SOTA-inspired external export into the common evaluation
    # schema used by the portable dashboard and comparison tables.
    metrics_row = safe_read_csv(metrics_path).iloc[0].to_dict()
    predictions_df = safe_read_csv(predictions_path)
    prediction_rows = prediction_rows_from_frame(
        predictions_df,
        model_name="nephro_sota",
        variant=str(metrics_row.get("model_name", "nephro_sota_catboost")),
        dataset="external",
        true_col="label",
        pred_col="pred",
        score_col="prob",
        sample_id_col="canonical_smiles",
        sample_prefix="external",
    )

    metrics = compute_binary_metrics(
        true_labels=predictions_df["label"],
        predicted_labels=predictions_df["pred"],
        predicted_scores=predictions_df["prob"],
    )
    summary_row = make_summary_row(
        model_name="nephro_sota",
        variant=str(metrics_row.get("model_name", "nephro_sota_catboost")),
        dataset="external",
        accuracy=metrics_row.get("accuracy"),
        roc_auc=metrics_row.get("auroc"),
        notes=combine_notes(
            f"overlap_removed={metrics_row.get('overlap_removed')}",
            "detailed_predictions_available",
            "reused_archived_results" if reused_archived_results else None,
        ),
    )
    summary_row = merge_metrics(summary_row, metrics)
    return ModelResult(summary_rows=[summary_row], prediction_rows=prediction_rows)


def run_registered(config: dict[str, object]) -> ModelResult:
    # Rerun or reuse the SOTA-inspired CatBoost baseline from the legacy branch.
    force = bool(config.get("force_rerun", False))
    python_exec = config.get("python_executable")

    output_dir = DEFAULT_OUTPUT_DIR
    primary_metrics = output_dir / "external_metrics.csv"
    primary_predictions = output_dir / "external_predictions.csv"
    fallback_dir = LEGACY_CONFIRMED_DIR / "results_nephro_sota"

    if force or not (
        (primary_metrics.exists() and primary_predictions.exists())
        or ((fallback_dir / "external_metrics.csv").exists() and (fallback_dir / "external_predictions.csv").exists())
    ):
        run_python_script(
            Path(__file__),
            [
                "--train-path",
                str(DEFAULT_TRAIN_CSV),
                "--test-path",
                str(DEFAULT_EXTERNAL_CSV),
                "--output-dir",
                str(output_dir),
            ],
            cwd=PROJECT_ROOT,
            python_executable=str(python_exec) if python_exec else None,
        )

    metrics_path = locate_existing(primary_metrics, fallback_dir / "external_metrics.csv")
    predictions_path = locate_existing(primary_predictions, fallback_dir / "external_predictions.csv")
    return _collect_registered_result(
        metrics_path,
        predictions_path,
        reused_archived_results=metrics_path.parent == fallback_dir,
    )


if __name__ == "__main__":
    main()
