"""
scaffold_features.py
---------------------
Extracts scaffold-based features from the provided scaffold CSVs.

This is a NOVEL feature set that Liu et al. (2025) never used.
The paper performed scaffold analysis only for diversity assessment —
we convert it directly into predictive features.

Features created per molecule
------------------------------
  murcko_scaffold          str    — Bemis-Murcko scaffold SMILES
  in_toxic_scaffold_top50  int    — 1 if scaffold appears in top-50 most
                                    frequent nephrotoxic scaffolds
  toxic_scaffold_freq      float  — frequency of this scaffold in toxic set
                                    (0 if not seen)
  in_model_scaffold        int    — 1 if scaffold seen in any training molecule
  in_ext_scaffold          int    — 1 if scaffold seen in external test set
                                    (useful signal for novelty-aware weighting)
  scaffold_overlap_flag    int    — 1 if scaffold is in both train AND external
  carbon_scaffold_freq     float  — frequency of carbon scaffold in training set
  n_fused_rings            int    — number of fused ring systems (from RDKit)
  scaffold_mw_bin          int    — molecular weight bucket of the scaffold (0-4)

Why these features help
-----------------------
  1. Toxic scaffolds identified from 1527 training molecules encode
     domain knowledge about structural alerts at zero extra annotation cost.
  2. A molecule sharing a scaffold with many known toxics is more
     likely to be toxic — this is essentially scaffold-hop-aware
     nearest-neighbour information.
  3. The overlap/novelty flag lets the model hedge on chemotypes it
     has not seen before — complementing the Tanimoto GP uncertainty.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError("rdkit is required:  pip install rdkit")


# ---------------------------------------------------------------------------
# Scaffold extraction helpers
# ---------------------------------------------------------------------------

def _get_murcko(smi: str) -> str:
    """Return Bemis-Murcko scaffold SMILES, or '' on failure."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def _n_fused_rings(smi: str) -> int:
    """Count fused ring systems."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0
        ri = mol.GetRingInfo()
        return len(ri.AtomRings())
    except Exception:
        return 0


def _scaffold_mw_bin(scaffold_smi: str) -> int:
    """Bucket scaffold molecular weight: 0=<100, 1=100-200, 2=200-300, 3=300-400, 4=400+"""
    try:
        mol = Chem.MolFromSmiles(scaffold_smi)
        if mol is None:
            return 0
        mw = Descriptors.MolWt(mol)
        return min(int(mw // 100), 4)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ScaffoldFeatureExtractor:
    """
    Builds scaffold-based features using the scaffold CSVs from Liu et al.

    Parameters
    ----------
    scaffold_csvs : dict
        Mapping from key to CSV path, as defined in config.SCAFFOLD_CSVs.
    top_n_toxic : int, default 50
        Number of most-frequent toxic scaffolds to use for binary flag.
    """

    def __init__(self, scaffold_csvs: Dict[str, Path], top_n_toxic: int = 50):
        self.scaffold_csvs = scaffold_csvs
        self.top_n_toxic   = top_n_toxic

        # These are set during fit()
        self._toxic_scaffold_freq:  Dict[str, float] = {}
        self._top_toxic_scaffolds:  set               = set()
        self._model_scaffolds:      set               = set()
        self._ext_scaffolds:        set               = set()
        self._carbon_scaffold_freq: Dict[str, float]  = {}
        self._fitted = False

    def fit(self, smiles_train: pd.Series) -> "ScaffoldFeatureExtractor":
        """
        Fit the extractor using the scaffold CSVs and the training SMILES.
        """
        logger.info("Fitting scaffold feature extractor ...")

        # Load toxic Murcko scaffold frequencies
        tox_df = pd.read_csv(self.scaffold_csvs["murcko_tox"])
        tox_df = tox_df[["scaffold", "frequency"]].dropna()
        total_tox = tox_df["frequency"].sum()
        self._toxic_scaffold_freq = {
            row["scaffold"]: row["frequency"] / total_tox
            for _, row in tox_df.iterrows()
        }
        top = tox_df.nlargest(self.top_n_toxic, "frequency")["scaffold"].tolist()
        self._top_toxic_scaffolds = set(top)

        # All model scaffolds
        model_df = pd.read_csv(self.scaffold_csvs["murcko_model"])
        self._model_scaffolds = set(model_df["scaffold"].dropna())

        # External test scaffolds
        ext_df = pd.read_csv(self.scaffold_csvs["murcko_ext"])
        self._ext_scaffolds = set(ext_df["scaffold"].dropna())

        # Carbon scaffold frequencies
        carbon_df = pd.read_csv(self.scaffold_csvs["carbon_model"])
        total_c = carbon_df["frequency"].sum()
        self._carbon_scaffold_freq = {
            row["scaffold"]: row["frequency"] / total_c
            for _, row in carbon_df[["scaffold", "frequency"]].dropna().iterrows()
        }

        logger.info(
            f"  Toxic scaffolds loaded: {len(self._toxic_scaffold_freq)} | "
            f"Top-{self.top_n_toxic} toxic: {len(self._top_toxic_scaffolds)} | "
            f"Model scaffolds: {len(self._model_scaffolds)} | "
            f"External scaffolds: {len(self._ext_scaffolds)}"
        )
        self._fitted = True
        return self

    def transform(self, smiles_series: pd.Series) -> pd.DataFrame:
        """
        Transform a series of SMILES into a feature DataFrame.

        Returns
        -------
        DataFrame with columns:
            in_toxic_scaffold_top50, toxic_scaffold_freq,
            in_model_scaffold, in_ext_scaffold, scaffold_overlap_flag,
            carbon_scaffold_freq, n_fused_rings, scaffold_mw_bin
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        rows = []
        for smi in smiles_series:
            scaffold = _get_murcko(str(smi))

            in_top50   = int(scaffold in self._top_toxic_scaffolds)
            tox_freq   = self._toxic_scaffold_freq.get(scaffold, 0.0)
            in_model   = int(scaffold in self._model_scaffolds)
            in_ext     = int(scaffold in self._ext_scaffolds)
            overlap    = int(in_model and in_ext)
            carbon_f   = self._carbon_scaffold_freq.get(scaffold, 0.0)
            fused      = _n_fused_rings(str(smi))
            mw_bin     = _scaffold_mw_bin(scaffold)

            rows.append({
                "in_toxic_scaffold_top50": in_top50,
                "toxic_scaffold_freq":     tox_freq,
                "in_model_scaffold":       in_model,
                "in_ext_scaffold":         in_ext,
                "scaffold_overlap_flag":   overlap,
                "carbon_scaffold_freq":    carbon_f,
                "n_fused_rings":           fused,
                "scaffold_mw_bin":         mw_bin,
            })

        df = pd.DataFrame(rows, index=smiles_series.index)
        logger.info(
            f"Scaffold features: {df.shape[1]} cols x {df.shape[0]} rows | "
            f"top50 hits: {df['in_toxic_scaffold_top50'].sum()} "
            f"({100*df['in_toxic_scaffold_top50'].mean():.1f}%)"
        )
        return df

    def fit_transform(self, smiles_series: pd.Series) -> pd.DataFrame:
        return self.fit(smiles_series).transform(smiles_series)
