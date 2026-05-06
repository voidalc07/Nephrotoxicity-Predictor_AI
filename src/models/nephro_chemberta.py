import argparse
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
DEFAULT_OUTPUT_DIR = RAW_RUNS_DIR / "nephro_chemberta"
INTERNAL_CV_PATH = DEFAULT_OUTPUT_DIR / "internal_cv_metrics.csv"

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
    # The ChemBERTa hybrid can run either against the bundled portable datasets
    # or against an explicitly supplied legacy data directory. This preserves
    # portability while remaining faithful to the original benchmark assets.
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
    raise FileNotFoundError(
        "Could not find a data directory. Pass --data-dir explicitly."
    )


def pick_smiles_column(df: pd.DataFrame) -> str:
    # Historical datasets use slightly different column names for canonicalised
    # SMILES, so the loader normalises common alternatives here.
    normalized = {c.strip().lower().replace("_", " "): c for c in df.columns}
    for key in ("canonical smiles", "canonical_smiles", "smiles", "smiles clean"):
        if key in normalized:
            return normalized[key]
    raise ValueError("No SMILES-like column found in dataset.")


def get_rdkit_features(smiles: str) -> dict[str, float] | None:
    # -------------------------------------------------------------------------
    # RDKit Descriptor + Fingerprint + Alert Featurisation
    # This branch augments transformer embeddings with classical cheminformatics
    # variables. RDKit descriptors encode physicochemical properties, Morgan
    # fingerprints capture circular substructures, and SMARTS alerts inject a
    # hand-crafted toxicophore view motivated by mechanistic nephrotoxicity
    # patterns described in the project methodology.
    # -------------------------------------------------------------------------
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


def load_chemberta(model_name: str, device: torch.device) -> tuple[AutoTokenizer, AutoModel]:
    # -------------------------------------------------------------------------
    # Transformer Backbone Loading
    # ChemBERTa is used as a pretrained molecular language model over SMILES.
    # The loader prefers offline assets for dissertation reproducibility, but
    # can still fall back to Hugging Face downloads if a local cache is absent.
    # -------------------------------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=False,
        ).to(device)
        return tokenizer, model
    except Exception:
        # If the local copy is missing, try downloading it instead.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, use_safetensors=False).to(device)
        return tokenizer, model


def get_chemberta_embeddings(
    smiles_list: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    # CLS-token embeddings are extracted batchwise so each molecule gains a
    # dense learned representation that complements the handcrafted RDKit view.
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)


def load_data(
    path: Path,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # -------------------------------------------------------------------------
    # Hybrid Feature Assembly
    # The final ChemBERTa feature matrix is a horizontal concatenation of
    # handcrafted chemistry features and transformer embeddings. Architecturally
    # this keeps the learned-sequence and tabular-information pathways aligned
    # in a single model-ready tensor.
    # -------------------------------------------------------------------------
    df = pd.read_csv(path)
    smiles_col = pick_smiles_column(df)

    features, labels, valid_smiles = [], [], []
    for _, row in df.iterrows():
        f = get_rdkit_features(str(row[smiles_col]))
        if f is not None:
            features.append(f)
            labels.append(int(row["label"]))
            valid_smiles.append(str(row[smiles_col]))

    df_features = pd.DataFrame(features).fillna(0)
    cb_emb = get_chemberta_embeddings(valid_smiles, tokenizer, model, device, batch_size, max_length)
    return np.hstack((df_features.values, cb_emb)), np.array(labels), valid_smiles


def pick_device(choice: str) -> torch.device:
    # Device selection is explicit because transformer inference can be run on
    # CPU for portability or on Apple MPS where available for faster reruns.
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    # Use MPS when it is available, otherwise fall back to CPU.
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _compute_internal_cv_metrics(
    *,
    train_path: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    iterations: int,
    threads: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    # -------------------------------------------------------------------------
    # Internal Cross-Validation Estimate
    # The archived ChemBERTa export preserved external metrics but not a full
    # internal-CV table for the portable dashboard. This routine therefore
    # regenerates a 5-fold estimate using the same hybrid formulation so the
    # analytics page can report a generalisation gap.
    #
    # NOTE:
    # These internal metrics are regenerated estimates, not original archived
    # benchmark files from the legacy project.
    # -------------------------------------------------------------------------
    tokenizer, chemberta = load_chemberta(model_name, device)
    chemberta.eval()
    X, y, _ = load_data(train_path, tokenizer, chemberta, device, batch_size, max_length)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y), dtype=float)

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_valid = X[valid_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_valid_s = scaler.transform(X_valid)

        cb = CatBoostClassifier(
            iterations=iterations,
            verbose=False,
            thread_count=threads,
            random_seed=seed + fold_index,
        )
        cb.fit(X_train_s, y_train)
        cb_probs = cb.predict_proba(X_valid_s)[:, 1]

        lgbm = LGBMClassifier(
            n_estimators=iterations,
            n_jobs=threads,
            random_state=seed + fold_index,
            verbose=-1,
        )
        lgbm.fit(X_train_s, y_train)
        lgbm_probs = lgbm.predict_proba(X_valid_s)[:, 1]

        oof_probs[valid_idx] = np.mean(np.column_stack([cb_probs, lgbm_probs]), axis=1)

    preds = (oof_probs >= 0.5).astype(int)
    metrics = compute_binary_metrics(
        true_labels=y,
        predicted_labels=preds,
        predicted_scores=oof_probs,
    )
    return pd.DataFrame(
        [
            {
                "model_name": "chemberta_hybrid_cb_lgbm",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auroc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "notes": "generated_internal_cv_estimate_5fold",
            }
        ]
    )


def main() -> None:
    # -------------------------------------------------------------------------
    # ChemBERTa Hybrid Benchmark Script
    # The experimental design pairs ChemBERTa embeddings with CatBoost and
    # LightGBM, then averages their probabilities. CatBoost is well-suited to
    # dense heterogeneous tabular features, while LightGBM remains efficient on
    # wider feature spaces. The simple mean ensemble keeps inference stable
    # without introducing a further meta-learner in this branch.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Stable ChemBERTa hybrid baseline.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-path", default=None, help="Optional explicit train CSV path.")
    parser.add_argument("--test-path", default=None, help="Optional explicit external test CSV path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", default="DeepChem/ChemBERTa-77M-MTR")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    data_dir = resolve_data_dir(args.data_dir)
    train_path = Path(args.train_path) if args.train_path else data_dir / "model construction dataset.csv"
    test_path = Path(args.test_path) if args.test_path else data_dir / "external test dataset.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    print(f"Using device: {device}")
    print(f"Train CSV: {train_path}")
    print(f"Test CSV:  {test_path}")

    tokenizer, chemberta = load_chemberta(args.model_name, device)
    chemberta.eval()

    print("Extracting ChemBERTa + RDKit features...")
    X_train, y_train, _ = load_data(
        train_path,
        tokenizer,
        chemberta,
        device,
        args.batch_size,
        args.max_length,
    )
    X_test, y_test, test_smiles = load_data(
        test_path,
        tokenizer,
        chemberta,
        device,
        args.batch_size,
        args.max_length,
    )

    # Standardisation is applied after feature fusion because the transformer
    # embedding dimensions and RDKit descriptors operate on very different
    # numeric scales.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training CatBoost...")
    # CatBoost offers strong default handling for dense feature matrices and is
    # retained here as one of the two complementary learners in the hybrid.
    cb = CatBoostClassifier(
        iterations=args.iterations,
        verbose=False,
        thread_count=args.threads,
        random_seed=args.seed,
    )
    cb.fit(X_train_s, y_train)
    cb_probs = cb.predict_proba(X_test_s)[:, 1]

    print("Training LightGBM...")
    # LightGBM supplies a second inductive bias with efficient boosted trees.
    lgbm = LGBMClassifier(
        n_estimators=args.iterations,
        n_jobs=args.threads,
        random_state=args.seed,
        verbose=-1,
    )
    lgbm.fit(X_train_s, y_train)
    lgbm_probs = lgbm.predict_proba(X_test_s)[:, 1]

    # ---------------------------------------------------------------------
    # Probability Ensembling
    # A simple arithmetic mean is used as the final hybrid score. This keeps
    # the branch interpretable and low-friction, while still exploiting the
    # complementary behaviour of CatBoost and LightGBM on the fused features.
    # ---------------------------------------------------------------------
    # Combine the model scores that are working reliably here.
    probs = np.mean(np.column_stack([cb_probs, lgbm_probs]), axis=1)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "model_name": "chemberta_hybrid_cb_lgbm",
        "auroc": float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "kappa": float(cohen_kappa_score(y_test, preds)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    print("\n--- Final Results on External Set ---")
    print(f"AUROC:    {metrics['auroc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1:       {metrics['f1']:.4f}")
    print(f"Kappa:    {metrics['kappa']:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(out_dir / "external_metrics.csv", index=False)
    pd.DataFrame(
        {
            "canonical_smiles": test_smiles,
            "label": y_test,
            "prob": probs,
            "pred": preds,
        }
    ).to_csv(out_dir / "external_predictions.csv", index=False)

    print(f"Saved outputs to: {out_dir.resolve()}")


def _collect_registered_result(
    metrics_path: Path,
    predictions_path: Path,
    internal_metrics_path: Path | None = None,
    *,
    reused_archived_results: bool,
) -> ModelResult:
    # Repackage the saved CSV artefacts into the shared evaluation schema used
    # by the consolidated benchmarking and dashboard layers.
    metrics_row = safe_read_csv(metrics_path).iloc[0].to_dict()
    predictions_df = safe_read_csv(predictions_path)
    prediction_rows = prediction_rows_from_frame(
        predictions_df,
        model_name="nephro_chemberta",
        variant=str(metrics_row.get("model_name", "chemberta_hybrid_cb_lgbm")),
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
        model_name="nephro_chemberta",
        variant=str(metrics_row.get("model_name", "chemberta_hybrid_cb_lgbm")),
        dataset="external",
        accuracy=metrics_row.get("accuracy"),
        f1=metrics_row.get("f1"),
        roc_auc=metrics_row.get("auroc"),
        notes=combine_notes(
            "detailed_predictions_available",
            "reused_archived_results" if reused_archived_results else None,
        ),
    )
    summary_row = merge_metrics(summary_row, metrics)
    summary_rows = [summary_row]

    if internal_metrics_path is not None and internal_metrics_path.exists():
        internal_row = safe_read_csv(internal_metrics_path).iloc[0].to_dict()
        summary_rows.append(
            make_summary_row(
                model_name="nephro_chemberta",
                variant=str(internal_row.get("model_name", "chemberta_hybrid_cb_lgbm")),
                dataset="internal_cv",
                accuracy=internal_row.get("accuracy"),
                precision=internal_row.get("precision"),
                recall=internal_row.get("recall"),
                f1=internal_row.get("f1"),
                roc_auc=internal_row.get("auroc"),
                pr_auc=internal_row.get("pr_auc"),
                notes=combine_notes(
                    internal_row.get("notes"),
                    "reused_archived_results" if reused_archived_results else None,
                ),
            )
        )

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def run_registered(config: dict[str, object]) -> ModelResult:
    # -------------------------------------------------------------------------
    # Portable Registration Wrapper
    # This function either reuses archived ChemBERTa results or regenerates
    # them locally, then augments them with an internal-CV estimate so the
    # model can appear alongside the other engines in the analytics dashboard.
    # -------------------------------------------------------------------------
    force = bool(config.get("force_rerun", False))
    python_exec = config.get("python_executable")

    output_dir = DEFAULT_OUTPUT_DIR
    primary_metrics = output_dir / "external_metrics.csv"
    primary_predictions = output_dir / "external_predictions.csv"
    fallback_dir = LEGACY_CONFIRMED_DIR / "results_nephro_chemberta"
    internal_metrics_path = output_dir / "internal_cv_metrics.csv"

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
    if force or not internal_metrics_path.exists():
        internal_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        internal_df = _compute_internal_cv_metrics(
            train_path=DEFAULT_TRAIN_CSV,
            model_name="DeepChem/ChemBERTa-77M-MTR",
            batch_size=64,
            max_length=256,
            iterations=500,
            threads=1,
            seed=42,
            device=torch.device("cpu"),
        )
        internal_df.to_csv(internal_metrics_path, index=False)
    return _collect_registered_result(
        metrics_path,
        predictions_path,
        internal_metrics_path,
        reused_archived_results=metrics_path.parent == fallback_dir,
    )


if __name__ == "__main__":
    main()
