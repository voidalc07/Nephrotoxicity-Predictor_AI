"""
featurizer.py — ChemBERTa-2 + RDKit descriptors + ECFP6 + scaffold features.
Used by all ML-based models (LightGBM, HistGB, GPC, NODE, Ensemble).
GIN has its own internal graph featurizer.
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError("pip install rdkit")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scaffold_features import ScaffoldFeatureExtractor

_DESC_NAMES = [
    "MolWt","ExactMolWt","HeavyAtomCount","NumHAcceptors","NumHDonors",
    "NumHeteroatoms","NumRotatableBonds","NumAromaticRings","NumSaturatedRings",
    "NumAliphaticRings","RingCount","FractionCSP3","TPSA","MolLogP","MolMR",
    "BertzCT","Ipc","Kappa1","Kappa2","Kappa3","Chi0n","Chi1n","Chi2n",
    "Chi0v","Chi1v","NHOHCount","NOCount","NumRadicalElectrons","NumValenceElectrons",
    "BalabanJ","HallKierAlpha","LabuteASA","PEOE_VSA1","PEOE_VSA2","PEOE_VSA3",
    "PEOE_VSA4","PEOE_VSA5","SMR_VSA1","SMR_VSA2","SlogP_VSA1","SlogP_VSA2",
    "EState_VSA1","EState_VSA2","MaxEStateIndex","MinEStateIndex","qed",
]
_DESC_FUNCS = {n: getattr(Descriptors, n) for n in _DESC_NAMES if hasattr(Descriptors, n)}


def _parse(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        return mol if (mol is not None and mol.GetNumAtoms() > 0) else None
    except Exception: return None

def _ecfp6(mol, n_bits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits)
    return np.frombuffer(fp.ToBitString().encode(), dtype="u1") - ord("0")

def _desc(mol):
    v = []
    for fn in _DESC_FUNCS.values():
        try: v.append(float(fn(mol) or np.nan))
        except Exception: v.append(np.nan)
    return np.array(v, dtype=np.float32)

def _load_chemberta(primary, fallback):
    from transformers import AutoTokenizer, AutoModel
    import torch
    for name in [primary, fallback]:
        try:
            logger.info(f"Loading ChemBERTa-2: {name} ...")
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name); mdl.eval()
            logger.info(f"  Loaded: {name}")
            return tok, mdl, torch
        except Exception as e:
            logger.warning(f"  Failed ({name}): {e}")
    raise RuntimeError("Could not load any ChemBERTa-2 model.")


class MolecularFeaturizer:
    """
    Produces:
      X_combined  (n, ~1800)  — [BERT | RDKit | ECFP6 | scaffold]  for LightGBM/HistGB/NODE
      X_tabpfn    (n, 150)    — PCA(X_combined)                     for TabNet/NODE
      X_fp        (n, 1024)   — raw binary fingerprints              for Tanimoto GPC
    """
    def __init__(self, scaffold_csvs, fp_bits=1024, pca_components=150,
                 bert_batch=32,
                 bert_model="DeepChem/ChemBERTa-77M-MTR",
                 bert_fallback="seyonec/ChemBERTa-zinc-base-v2",
                 seed=42):
        self.scaffold_csvs  = scaffold_csvs
        self.fp_bits        = fp_bits
        self.pca_components = pca_components
        self.bert_batch     = bert_batch
        self.bert_model     = bert_model
        self.bert_fallback  = bert_fallback
        self.seed           = seed
        self._tok = self._mdl = self._torch = None
        self._scaff  = ScaffoldFeatureExtractor(scaffold_csvs)
        self._imputer = SimpleImputer(strategy="median")
        self._scaler  = StandardScaler()
        self._sscaler = StandardScaler()
        self._pca: Optional[PCA] = None
        self._fitted = False
        self.feature_names: List[str] = []

    def _ensure_bert(self):
        if self._tok is None:
            self._tok, self._mdl, self._torch = _load_chemberta(
                self.bert_model, self.bert_fallback)

    def _embed(self, smiles_list):
        self._ensure_bert()
        torch = self._torch
        out = []
        for i in range(0, len(smiles_list), self.bert_batch):
            batch = smiles_list[i:i+self.bert_batch]
            enc = self._tok(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            with torch.no_grad():
                h = self._mdl(**enc).last_hidden_state[:,0,:].cpu().numpy()
            out.append(h)
        return np.vstack(out).astype(np.float32)

    def _raw(self, smiles_list):
        fps, descs, valid = [], [], []
        for i, smi in enumerate(smiles_list):
            mol = _parse(smi)
            if mol is None:
                logger.warning(f"Row {i}: invalid SMILES — excluded.")
                valid.append(False)
                fps.append(np.zeros(self.fp_bits, dtype=np.float32))
                descs.append(np.full(len(_DESC_FUNCS), np.nan, dtype=np.float32))
            else:
                valid.append(True)
                fps.append(_ecfp6(mol, self.fp_bits).astype(np.float32))
                descs.append(_desc(mol))
        return np.stack(fps), np.stack(descs), np.array(valid, dtype=bool)

    def fit_transform(self, smiles_series: pd.Series):
        sml = smiles_series.tolist()
        fps, descs, mask = self._raw(sml)
        vi = np.where(mask)[0]
        n  = vi.shape[0]
        fps_v   = fps[mask];  descs_v = descs[mask]
        descs_i = self._imputer.fit_transform(descs_v)
        descs_s = self._scaler.fit_transform(descs_i)
        valid_smi = smiles_series.iloc[vi].reset_index(drop=True)
        scaff  = self._scaff.fit_transform(valid_smi)
        scaff_s = self._sscaler.fit_transform(scaff.values.astype(np.float32))
        logger.info(f"Computing BERT embeddings for {n} molecules ...")
        bert = self._embed(valid_smi.tolist())
        Xc = np.hstack([bert, descs_s, fps_v, scaff_s]).astype(np.float32)
        nc = min(self.pca_components, n-1, Xc.shape[1])
        self._pca = PCA(n_components=nc, random_state=self.seed)
        Xt = self._pca.fit_transform(Xc).astype(np.float32)
        self.feature_names = (
            [f"BERT_{i}" for i in range(bert.shape[1])]
            + list(_DESC_FUNCS.keys())
            + [f"FP_{i}" for i in range(self.fp_bits)]
            + list(scaff.columns)
        )
        self._fitted = True
        logger.info(f"Features: X_combined={Xc.shape} X_pca={Xt.shape} X_fp={fps_v.shape}")
        return Xc, Xt, fps_v, vi

    def transform(self, smiles_series: pd.Series):
        if not self._fitted: raise RuntimeError("Call fit_transform() first.")
        sml = smiles_series.tolist()
        fps, descs, mask = self._raw(sml)
        vi = np.where(mask)[0]
        fps_v   = fps[mask];  descs_v = descs[mask]
        descs_i = self._imputer.transform(descs_v)
        descs_s = self._scaler.transform(descs_i)
        valid_smi = smiles_series.iloc[vi].reset_index(drop=True)
        scaff   = self._scaff.transform(valid_smi)
        scaff_s = self._sscaler.transform(scaff.values.astype(np.float32))
        logger.info(f"Computing BERT embeddings for {len(valid_smi)} molecules ...")
        bert = self._embed(valid_smi.tolist())
        Xc = np.hstack([bert, descs_s, fps_v, scaff_s]).astype(np.float32)
        Xt = self._pca.transform(Xc).astype(np.float32)
        return Xc, Xt, fps_v, vi

    def save(self, path):
        _tok, _mdl, _torch = self._tok, self._mdl, self._torch
        self._tok = self._mdl = self._torch = None
        joblib.dump(self, path)
        self._tok = _tok; self._mdl = _mdl; self._torch = _torch
        logger.info(f"Featurizer saved → {path}")

    @classmethod
    def load(cls, path):
        f = joblib.load(path); logger.info(f"Featurizer loaded ← {path}"); return f
