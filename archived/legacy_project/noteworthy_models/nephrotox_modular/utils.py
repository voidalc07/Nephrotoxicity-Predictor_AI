"""utils.py — logging, data loading, and common helpers."""
import logging, sys
import numpy as np
import pandas as pd


def setup_logging(level=logging.INFO):
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)


def load_dataset(csv_path, smiles_col="canonical SMILES", label_col="label"):
    """Load CSV and normalise column names to 'smiles' and 'label'."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # Flexible SMILES column matching
    if smiles_col in df.columns:
        df = df.rename(columns={smiles_col: "smiles"})
    elif "smiles" not in df.columns:
        match = [c for c in df.columns if "smiles" in c.lower()]
        if match:
            df = df.rename(columns={match[0]: "smiles"})
        else:
            raise ValueError(f"No SMILES column found in {csv_path}. Columns: {list(df.columns)}")
    df["label"] = df[label_col].astype(int)
    return df.dropna(subset=["smiles", "label"]).reset_index(drop=True)[["smiles", "label"]]


def print_dist(y, name="Dataset", logger=None):
    n, pos = len(y), int(y.sum())
    msg = f"{name}: n={n} | Nephrotoxic(1)={pos}({100*pos/n:.1f}%) | Non-toxic(0)={n-pos}({100*(n-pos)/n:.1f}%)"
    (logger.info if logger else print)(msg)
