from __future__ import annotations

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional

from utils.logger import get_logger


logger = get_logger(__name__)


SMILES_CANDIDATES = [
    "canonical_smiles",
    "canonical smiles",
    "smiles",
    "smiles_clean",
]


def _normalize_colname(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_dataset(file_path: str, label_column: str = "label", require_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    normalized_to_original = {_normalize_colname(col): col for col in df.columns}

    smiles_col = None
    for cand in SMILES_CANDIDATES:
        key = _normalize_colname(cand)
        if key in normalized_to_original:
            smiles_col = normalized_to_original[key]
            break
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {file_path}; expected one of {SMILES_CANDIDATES}.")

    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    before = len(df)
    df = df.dropna(subset=["canonical_smiles"]).copy()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with invalid SMILES from %s", dropped, file_path)

    df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

    if require_labels:
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {file_path}.")
        df = df.dropna(subset=[label_column]).reset_index(drop=True)
        df[label_column] = df[label_column].astype(int)

    return df


def stratified_splits(
    df: pd.DataFrame,
    seed: int,
    label_column: str = "label",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0")

    temp_frac = train_frac + val_frac
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df[label_column],
        random_state=seed,
    )
    val_relative = val_frac / temp_frac
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative,
        stratify=df_train_val[label_column],
        random_state=seed,
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def compute_class_weights(labels: List[int]) -> List[float]:
    series = pd.Series(labels)
    counts = series.value_counts().to_dict()
    if len(counts) < 2:
        return [1.0, 1.0]
    total = sum(counts.values())
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    # catboost expects list ordered by class index
    return [weights.get(0, 1.0), weights.get(1, 1.0)]
