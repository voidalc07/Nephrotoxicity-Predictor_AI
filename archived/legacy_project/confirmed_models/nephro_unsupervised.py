import argparse
from pathlib import Path
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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

DEFAULT_ABS_DATA_DIR = Path("/Users/rajee/Desktop/KV6013-Induvidual Project/molecular data and scaffolds")
DEFAULT_REPO_DATA_DIR = Path(__file__).resolve().parents[1] / "noteworthy_models" / "nephrotox_fixed" / "data"

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
    normalized = {c.strip().lower().replace("_", " "): c for c in df.columns}
    for key in ("canonical smiles", "canonical_smiles", "smiles", "smiles clean"):
        if key in normalized:
            return normalized[key]
    raise ValueError("No SMILES-like column found in dataset.")


def get_features(smiles: str) -> list[int] | None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised representation + CatBoost baselines.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--output-dir", default="confirmed_models/results_nephro_unsupervised")
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


if __name__ == "__main__":
    main()
