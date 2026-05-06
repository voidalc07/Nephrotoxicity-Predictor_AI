"""
utils.py — Logging, validation, and general helpers.
"""

import logging
import sys
import numpy as np
import pandas as pd


def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level, format=fmt, datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )


def load_dataset(csv_path, smiles_col: str = "canonical SMILES",
                 label_col: str = "label") -> pd.DataFrame:
    """
    Load a dataset CSV and return a clean DataFrame with
    normalised column names: 'smiles' and 'label'.
    Handles the paper's 'canonical SMILES' column name automatically.
    """
    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Rename the SMILES column to a standard name
    if smiles_col in df.columns:
        df = df.rename(columns={smiles_col: "smiles"})
    elif "smiles" not in df.columns:
        # Try case-insensitive match
        matches = [c for c in df.columns if c.lower() == "smiles" or
                   "smiles" in c.lower()]
        if matches:
            df = df.rename(columns={matches[0]: "smiles"})
        else:
            raise ValueError(
                f"Could not find SMILES column in {csv_path}.\n"
                f"Available columns: {list(df.columns)}"
            )

    if label_col in df.columns:
        df["label"] = df[label_col].astype(int)
    elif "label" not in df.columns:
        raise ValueError(
            f"Could not find label column in {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["smiles", "label"]).reset_index(drop=True)
    return df[["smiles", "label"] + [c for c in df.columns
                                      if c not in ("smiles", "label")]]


def print_class_distribution(y: np.ndarray, name: str = "Dataset",
                              logger=None) -> None:
    n, pos = len(y), int(y.sum())
    neg = n - pos
    msg = (f"{name}: n={n} | "
           f"Nephrotoxic(1)={pos}({100*pos/n:.1f}%) | "
           f"Non-toxic(0)={neg}({100*neg/n:.1f}%)")
    (logger.info if logger else print)(msg)


def safe_n_folds(y: np.ndarray, requested: int) -> int:
    min_class = int(np.bincount(y.astype(int)).min())
    usable = min(requested, min_class)
    if usable < requested:
        logging.getLogger(__name__).warning(
            f"Minority class has {min_class} samples — "
            f"reducing folds {requested}→{usable}."
        )
    return max(usable, 2)
