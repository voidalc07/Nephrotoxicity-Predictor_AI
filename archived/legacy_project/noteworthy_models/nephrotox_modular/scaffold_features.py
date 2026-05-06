"""scaffold_features.py — novel scaffold-alert features from the paper's scaffold CSVs."""
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError("pip install rdkit")


def _get_murcko(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None: return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception: return ""

def _n_fused_rings(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None: return 0
        return len(mol.GetRingInfo().AtomRings())
    except Exception: return 0

def _scaffold_mw_bin(scaffold_smi):
    try:
        mol = Chem.MolFromSmiles(scaffold_smi)
        if mol is None: return 0
        return min(int(Descriptors.MolWt(mol) // 100), 4)
    except Exception: return 0


class ScaffoldFeatureExtractor:
    def __init__(self, scaffold_csvs: Dict[str, Path], top_n_toxic: int = 50):
        self.scaffold_csvs = scaffold_csvs
        self.top_n_toxic   = top_n_toxic
        self._toxic_freq:  dict = {}
        self._top_toxic:   set  = set()
        self._model_scaffolds: set = set()
        self._ext_scaffolds:   set = set()
        self._carbon_freq: dict = {}
        self._fitted = False

    def fit(self, smiles_series: pd.Series) -> "ScaffoldFeatureExtractor":
        tox = pd.read_csv(self.scaffold_csvs["murcko_tox"])[["scaffold","frequency"]].dropna()
        total_t = tox["frequency"].sum()
        self._toxic_freq = {r.scaffold: r.frequency / total_t for r in tox.itertuples()}
        self._top_toxic  = set(tox.nlargest(self.top_n_toxic, "frequency")["scaffold"])
        self._model_scaffolds = set(pd.read_csv(self.scaffold_csvs["murcko_model"])["scaffold"].dropna())
        self._ext_scaffolds   = set(pd.read_csv(self.scaffold_csvs["murcko_ext"])["scaffold"].dropna())
        carbon = pd.read_csv(self.scaffold_csvs["carbon_model"])[["scaffold","frequency"]].dropna()
        tc = carbon["frequency"].sum()
        self._carbon_freq = {r.scaffold: r.frequency / tc for r in carbon.itertuples()}
        logger.info(f"Scaffold extractor fitted: {len(self._toxic_freq)} toxic scaffolds, "
                    f"top-{self.top_n_toxic}: {len(self._top_toxic)}")
        self._fitted = True
        return self

    def transform(self, smiles_series: pd.Series) -> pd.DataFrame:
        rows = []
        for smi in smiles_series:
            sc = _get_murcko(str(smi))
            rows.append({
                "in_toxic_scaffold_top50": int(sc in self._top_toxic),
                "toxic_scaffold_freq":     self._toxic_freq.get(sc, 0.0),
                "in_model_scaffold":       int(sc in self._model_scaffolds),
                "in_ext_scaffold":         int(sc in self._ext_scaffolds),
                "scaffold_overlap_flag":   int((sc in self._model_scaffolds) and (sc in self._ext_scaffolds)),
                "carbon_scaffold_freq":    self._carbon_freq.get(sc, 0.0),
                "n_fused_rings":           _n_fused_rings(str(smi)),
                "scaffold_mw_bin":         _scaffold_mw_bin(sc),
            })
        return pd.DataFrame(rows, index=smiles_series.index)

    def fit_transform(self, smiles_series: pd.Series) -> pd.DataFrame:
        return self.fit(smiles_series).transform(smiles_series)
