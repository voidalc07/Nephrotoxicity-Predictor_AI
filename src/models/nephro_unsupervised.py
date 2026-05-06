import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostClassifier
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

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
DEFAULT_OUTPUT_DIR = RAW_RUNS_DIR / "nephro_unsupervised"
INTERNAL_CV_PATH = DEFAULT_OUTPUT_DIR / "internal_cv_metrics.csv"
MAIN_VARIANT = "Autoencoder"
PROTOTYPE_VARIANT = "PCA"

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
    # The unsupervised baseline reuses the same benchmark datasets as the other
    # portable runners so representation-learning performance can be compared
    # directly against descriptor, fingerprint, and similarity models.
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
    # Support minor schema variations across archived CSV exports.
    normalized = {c.strip().lower().replace("_", " "): c for c in df.columns}
    for key in ("canonical smiles", "canonical_smiles", "smiles", "smiles clean"):
        if key in normalized:
            return normalized[key]
    raise ValueError("No SMILES-like column found in dataset.")


def get_features(smiles: str) -> list[int] | None:
    # -------------------------------------------------------------------------
    # Sparse Structural Input Space
    # This branch deliberately uses Morgan fingerprints plus SMARTS alert bits
    # as the autoencoder input. The aim is to learn a compressed latent view of
    # the same local structural environments and toxicophore cues used elsewhere
    # in the project, rather than introducing an entirely unrelated feature set.
    # -------------------------------------------------------------------------
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem import rdFingerprintGenerator

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = list(mfpgen.GetFingerprint(mol))
    alerts = []
    for smarts in ALERTS.values():
        patt = Chem.MolFromSmarts(smarts)
        alerts.append(1 if (patt and mol.HasSubstructMatch(patt)) else 0)
    return fp + alerts


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Build the model matrix while preserving the canonical SMILES strings used
    # for later external prediction exports and dashboard lookup.
    df = pd.read_csv(path)
    smiles_col = pick_smiles_column(df)
    features, labels, smiles_out = [], [], []
    for _, row in df.iterrows():
        f = get_features(str(row[smiles_col]))
        if f is not None:
            features.append(f)
            labels.append(int(row["label"]))
            smiles_out.append(str(row[smiles_col]))
    return np.array(features), np.array(labels), smiles_out


class Autoencoder(nn.Module):
    # -------------------------------------------------------------------------
    # Symmetric Reconstruction Network
    # The encoder compresses high-dimensional sparse chemistry vectors into a
    # dense latent representation, while the decoder reconstructs the original
    # fingerprint/alert space. This follows the unsupervised representation-
    # learning rationale described in the project methodology.
    # -------------------------------------------------------------------------
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)


def run_catboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    label: str,
    iterations: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    # CatBoost is used downstream of both PCA and autoencoder embeddings so the
    # comparison focuses on representation quality rather than classifier choice.
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        verbose=False,
        thread_count=-1,
        random_seed=seed,
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = model.predict(X_te).astype(int)

    metrics = {
        "representation": label,
        "auroc": float(roc_auc_score(y_te, probs)),
        "accuracy": float(accuracy_score(y_te, preds)),
        "f1": float(f1_score(y_te, preds)),
        "kappa": float(cohen_kappa_score(y_te, preds)),
    }
    print(f"[{label}] AUROC: {metrics['auroc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return metrics, probs


def _catboost_probabilities(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    # Shared helper for generating out-of-fold probabilities during internal CV.
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        verbose=False,
        thread_count=-1,
        random_seed=seed,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]


def _autoencoder_embeddings(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    # -------------------------------------------------------------------------
    # Latent Embedding Extraction
    # Reconstruction loss encourages the encoder to retain the most informative
    # combinatorial structure in the sparse chemical input while suppressing
    # redundancy. The encoded vectors are then used as compact supervised
    # features for the downstream CatBoost classifier.
    # -------------------------------------------------------------------------
    input_dim = X_train.shape[1]
    ae_model = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        ae_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            recon = ae_model(batch[0])
            loss = criterion(recon, batch[0])
            loss.backward()
            optimizer.step()

    ae_model.eval()
    with torch.no_grad():
        train_emb = ae_model.encoder(X_train_t).cpu().numpy()
        valid_emb = ae_model.encoder(X_valid_t).cpu().numpy()
    return train_emb, valid_emb


def _compute_internal_cv_metrics(
    *,
    train_path: Path,
    seed: int = 42,
    iterations: int = 1000,
    ae_epochs: int = 100,
    ae_latent_dim: int = 64,
    ae_batch_size: int = 64,
    device: torch.device | None = None,
) -> pd.DataFrame:
    # -------------------------------------------------------------------------
    # Internal CV for PCA vs Autoencoder
    # The unsupervised family was archived mainly with external metrics. This
    # helper reconstructs a matched 5-fold internal estimate for both the PCA
    # prototype and the neural autoencoder branch so their transfer gap can be
    # shown consistently on the analytics page.
    #
    # NOTE:
    # These internal values are regenerated portable estimates rather than
    # original archived leaderboard exports.
    # -------------------------------------------------------------------------
    if device is None:
        device = torch.device("cpu")

    X, y, _ = load_data(train_path)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    pca_oof = np.zeros(len(y), dtype=float)
    ae_oof = np.zeros(len(y), dtype=float)

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_valid = X[valid_idx]

        pca = PCA(n_components=min(50, X_train.shape[0], X_train.shape[1]))
        X_train_pca = pca.fit_transform(X_train)
        X_valid_pca = pca.transform(X_valid)
        pca_oof[valid_idx] = _catboost_probabilities(
            X_train_pca,
            y_train,
            X_valid_pca,
            iterations=iterations,
            seed=seed + fold_index,
        )

        X_train_ae, X_valid_ae = _autoencoder_embeddings(
            X_train,
            X_valid,
            latent_dim=ae_latent_dim,
            batch_size=ae_batch_size,
            epochs=ae_epochs,
            device=device,
        )
        ae_oof[valid_idx] = _catboost_probabilities(
            X_train_ae,
            y_train,
            X_valid_ae,
            iterations=iterations,
            seed=seed + fold_index,
        )

    rows: list[dict[str, float | str]] = []
    for variant, probs in (("PCA", pca_oof), ("Autoencoder", ae_oof)):
        preds = (probs >= 0.5).astype(int)
        metrics = compute_binary_metrics(
            true_labels=y,
            predicted_labels=preds,
            predicted_scores=probs,
        )
        rows.append(
            {
                "representation": variant,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auroc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "notes": "generated_internal_cv_estimate_5fold",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # -------------------------------------------------------------------------
    # Unsupervised Representation Benchmark
    # The script compares two dimensionality-reduction philosophies on the same
    # structural input: linear PCA as a simple baseline, and a neural
    # autoencoder as a non-linear latent compressor. Both are evaluated with
    # the same CatBoost classifier to isolate the effect of representation.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Unsupervised representation + CatBoost baselines.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--ae-latent-dim", type=int, default=64)
    parser.add_argument("--ae-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    args = parser.parse_args()

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        device = torch.device("mps")
    elif args.device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    data_dir = resolve_data_dir(args.data_dir)
    train_path = Path(args.train_path) if args.train_path else data_dir / "model construction dataset.csv"
    test_path = Path(args.test_path) if args.test_path else data_dir / "external test dataset.csv"

    print("Extracting features (Morgan FP + Alerts)...")
    X_train, y_train, _ = load_data(train_path)
    X_test, y_test, test_smiles = load_data(test_path)

    print("\nApplying PCA (50 components)...")
    # PCA provides a transparent linear compression baseline against which the
    # non-linear autoencoder can be judged.
    pca = PCA(n_components=min(50, X_train.shape[0]))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_metrics, pca_probs = run_catboost(
        X_train_pca, y_train, X_test_pca, y_test, "PCA", args.iterations, args.seed
    )

    print("\nTraining Autoencoder on Training Set...")
    input_dim = X_train.shape[1]
    ae_model = Autoencoder(input_dim, args.ae_latent_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=args.ae_batch_size, shuffle=True)

    for _ in range(args.ae_epochs):
        ae_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            recon = ae_model(batch[0])
            loss = criterion(recon, batch[0])
            loss.backward()
            optimizer.step()

    ae_model.eval()
    with torch.no_grad():
        X_train_ae = ae_model.encoder(X_train_t).cpu().numpy()
        X_test_ae = ae_model.encoder(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    ae_metrics, ae_probs = run_catboost(
        X_train_ae, y_train, X_test_ae, y_test, "Autoencoder", args.iterations, args.seed
    )

    print("\n--- Summary ---")
    print(f"PCA AUROC:         {pca_metrics['auroc']:.4f}")
    print(f"Autoencoder AUROC: {ae_metrics['auroc']:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([pca_metrics, ae_metrics]).to_csv(out_dir / "external_metrics.csv", index=False)
    pd.DataFrame(
        {
            "canonical_smiles": test_smiles,
            "label": y_test,
            "prob_pca": pca_probs,
            "pred_pca": (pca_probs >= 0.5).astype(int),
            "prob_autoencoder": ae_probs,
            "pred_autoencoder": (ae_probs >= 0.5).astype(int),
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
    # Rehydrate both representation variants from the saved external predictions
    # so they can be filtered into either the main autoencoder engine or the
    # PCA prototype stream.
    metrics_df = safe_read_csv(metrics_path)
    predictions_df = safe_read_csv(predictions_path)

    variant_columns = {
        "PCA": ("prob_pca", "pred_pca"),
        "Autoencoder": ("prob_autoencoder", "pred_autoencoder"),
    }

    prediction_rows: list[dict[str, object]] = []
    for variant, (score_col, pred_col) in variant_columns.items():
        prediction_rows.extend(
            prediction_rows_from_frame(
                predictions_df,
                model_name="nephro_unsupervised",
                variant=variant,
                dataset="external",
                true_col="label",
                pred_col=pred_col,
                score_col=score_col,
                sample_id_col="canonical_smiles",
                sample_prefix="external",
            )
        )

    summary_rows: list[dict[str, object]] = []
    for _, raw_row in metrics_df.iterrows():
        row = raw_row.to_dict()
        variant = str(row.get("representation", "unsupervised"))
        score_col, pred_col = variant_columns[variant]
        metrics = compute_binary_metrics(
            true_labels=predictions_df["label"],
            predicted_labels=predictions_df[pred_col],
            predicted_scores=predictions_df[score_col],
        )
        summary_row = make_summary_row(
            model_name="nephro_unsupervised",
            variant=variant,
            dataset="external",
            accuracy=row.get("accuracy"),
            f1=row.get("f1"),
            roc_auc=row.get("auroc"),
            notes=combine_notes(
                "detailed_predictions_available",
                "reused_archived_results" if reused_archived_results else None,
            ),
        )
        summary_rows.append(merge_metrics(summary_row, metrics))

    if internal_metrics_path is not None and internal_metrics_path.exists():
        internal_df = safe_read_csv(internal_metrics_path)
        for _, raw_row in internal_df.iterrows():
            row = raw_row.to_dict()
            summary_rows.append(
                make_summary_row(
                    model_name="nephro_unsupervised",
                    variant=str(row.get("representation", "unsupervised")),
                    dataset="internal_cv",
                    accuracy=row.get("accuracy"),
                    precision=row.get("precision"),
                    recall=row.get("recall"),
                    f1=row.get("f1"),
                    roc_auc=row.get("auroc"),
                    pr_auc=row.get("pr_auc"),
                    notes=combine_notes(
                        row.get("notes"),
                        "reused_archived_results" if reused_archived_results else None,
                    ),
                )
            )

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def _select_variant(result: ModelResult, *, variant: str) -> ModelResult:
    # Variant filtering lets the portable project expose the autoencoder as a
    # main engine while retaining PCA as a traceable prototype.
    summary_rows = [row for row in result.summary_rows if row.get("variant") == variant]
    prediction_rows = [row for row in result.prediction_rows if row.get("variant") == variant]
    if not summary_rows:
        raise ValueError(f"Could not find unsupervised variant {variant!r} in registered results.")
    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def run_all_variants(config: dict[str, object]) -> ModelResult:
    # Orchestrate rerun-or-reuse behaviour for the unsupervised family and
    # ensure the supplemental internal-CV cache exists for dashboard analytics.
    force = bool(config.get("force_rerun", False))
    python_exec = config.get("python_executable")

    output_dir = DEFAULT_OUTPUT_DIR
    primary_metrics = output_dir / "external_metrics.csv"
    primary_predictions = output_dir / "external_predictions.csv"
    fallback_dir = LEGACY_CONFIRMED_DIR / "results_nephro_unsupervised"
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
            seed=42,
            iterations=1000,
            ae_epochs=100,
            ae_latent_dim=64,
            ae_batch_size=64,
            device=torch.device("cpu"),
        )
        internal_df.to_csv(internal_metrics_path, index=False)
    return _collect_registered_result(
        metrics_path,
        predictions_path,
        internal_metrics_path,
        reused_archived_results=metrics_path.parent == fallback_dir,
    )


def run_registered(config: dict[str, object]) -> ModelResult:
    # Return the dissertation-facing autoencoder branch and mark it explicitly
    # as the selected main variant.
    result = _select_variant(run_all_variants(config), variant=MAIN_VARIANT)
    for row in result.summary_rows:
        row["notes"] = combine_notes(row.get("notes"), "selected_main_variant")
    return result


def run_pca_prototype(config: dict[str, object]) -> ModelResult:
    # Expose the PCA baseline as a prototype model for historical comparison.
    return _select_variant(run_all_variants(config), variant=PROTOTYPE_VARIANT)


if __name__ == "__main__":
    main()
