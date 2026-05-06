import argparse
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
    raise FileNotFoundError(
        "Could not find a data directory. Pass --data-dir explicitly."
    )


def pick_smiles_column(df: pd.DataFrame) -> str:
    normalized = {c.strip().lower().replace("_", " "): c for c in df.columns}
    for key in ("canonical smiles", "canonical_smiles", "smiles", "smiles clean"):
        if key in normalized:
            return normalized[key]
    raise ValueError("No SMILES-like column found in dataset.")


def get_rdkit_features(smiles: str) -> dict[str, float] | None:
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=False,
        ).to(device)
        return tokenizer, model
    except Exception:
        # Fallback when cache is empty but network is available.
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
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    # auto
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable ChemBERTa hybrid baseline.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-path", default=None, help="Optional explicit train CSV path.")
    parser.add_argument("--test-path", default=None, help="Optional explicit external test CSV path.")
    parser.add_argument("--output-dir", default="confirmed_models/results_nephro_chemberta")
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

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training CatBoost...")
    cb = CatBoostClassifier(
        iterations=args.iterations,
        verbose=False,
        thread_count=args.threads,
        random_seed=args.seed,
    )
    cb.fit(X_train_s, y_train)
    cb_probs = cb.predict_proba(X_test_s)[:, 1]

    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=args.iterations,
        n_jobs=args.threads,
        random_state=args.seed,
        verbose=-1,
    )
    lgbm.fit(X_train_s, y_train)
    lgbm_probs = lgbm.predict_proba(X_test_s)[:, 1]

    # Stability-first ensemble: average probabilities of models that are stable in this environment.
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


if __name__ == "__main__":
    main()
