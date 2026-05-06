"""
featurizer.py
-------------
Four feature blocks combined into one pipeline:

  Block A  X_bert    (n, 768)          ChemBERTa-2 CLS embeddings
  Block B  X_desc    (n, ~55)          RDKit 2D physicochemical descriptors
  Block C  X_fp      (n, 1024)         ECFP6 binary Morgan fingerprints
  Block D  X_scaff   (n, 8)            Scaffold-alert features (novel)

Combined for LightGBM:
  X_combined = [A | B | C | D]         (n, ~1855)

For TabPFN:
  X_tabpfn = PCA(X_combined, 100)      (n, 100)

For Tanimoto GP (raw bits only):
  X_fp                                  (n, 1024)
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
    raise ImportError("Install rdkit:  pip install rdkit")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from scaffold_features import ScaffoldFeatureExtractor


# ---------------------------------------------------------------------------
# RDKit descriptor list
# ---------------------------------------------------------------------------
_DESC_NAMES: List[str] = [
    "MolWt", "ExactMolWt", "HeavyAtomCount",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NumRotatableBonds", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "RingCount", "FractionCSP3",
    "TPSA", "MolLogP", "MolMR", "BertzCT", "Ipc",
    "Kappa1", "Kappa2", "Kappa3",
    "Chi0n", "Chi1n", "Chi2n", "Chi0v", "Chi1v",
    "NHOHCount", "NOCount", "NumRadicalElectrons", "NumValenceElectrons",
    "BalabanJ", "HallKierAlpha", "LabuteASA",
    "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3",
    "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
    "SMR_VSA1", "SMR_VSA2", "SMR_VSA3",
    "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3",
    "EState_VSA1", "EState_VSA2", "EState_VSA3",
    "MaxEStateIndex", "MinEStateIndex",
    "MaxAbsEStateIndex", "MinAbsEStateIndex",
    "qed",
]
_DESC_FUNCS = {n: getattr(Descriptors, n)
               for n in _DESC_NAMES if hasattr(Descriptors, n)}
_N_DESC = len(_DESC_FUNCS)


# ---------------------------------------------------------------------------
# Molecule-level helpers
# ---------------------------------------------------------------------------

def _parse(smi: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        return mol if (mol is not None and mol.GetNumAtoms() > 0) else None
    except Exception:
        return None


def _ecfp6(mol: Chem.Mol, n_bits: int = 1024) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=n_bits)
    return np.frombuffer(fp.ToBitString().encode(), dtype="u1") - ord("0")


def _rdkit_desc(mol: Chem.Mol) -> np.ndarray:
    vals = []
    for fn in _DESC_FUNCS.values():
        try:
            v = fn(mol)
            vals.append(float(v) if v is not None else np.nan)
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# ChemBERTa-2 loader
# ---------------------------------------------------------------------------

def _load_chemberta(primary: str, fallback: str):
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        raise ImportError("Install transformers and torch:\n"
                          "  pip install transformers torch")

    for name in [primary, fallback]:
        try:
            logger.info(f"Loading ChemBERTa-2: {name} ...")
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name)
            mdl.eval()
            logger.info(f"  Loaded: {name}")
            return tok, mdl, torch
        except Exception as e:
            logger.warning(f"  Failed ({name}): {e}")

    raise RuntimeError("Could not load any ChemBERTa-2 model. "
                       "Check your internet connection.")


# ---------------------------------------------------------------------------
# Main featurizer
# ---------------------------------------------------------------------------

class MolecularFeaturizer:
    """
    Full featurization pipeline.

    Parameters
    ----------
    scaffold_csvs : dict  — paths to scaffold CSVs (from config.SCAFFOLD_CSVs)
    fp_bits       : int   — ECFP6 bit count (default 1024)
    pca_components: int   — PCA dimensions for TabPFN input (default 100)
    bert_batch    : int   — SMILES per BERT forward pass (default 32)
    bert_model    : str   — HuggingFace model ID (primary)
    bert_fallback : str   — HuggingFace model ID (fallback)
    seed          : int
    """

    def __init__(
        self,
        scaffold_csvs: Dict,
        fp_bits: int = 1024,
        pca_components: int = 100,
        bert_batch: int = 32,
        bert_model: str = "DeepChem/ChemBERTa-77M-MTR",
        bert_fallback: str = "seyonec/ChemBERTa-zinc-base-v2",
        seed: int = 42,
    ):
        self.scaffold_csvs   = scaffold_csvs
        self.fp_bits         = fp_bits
        self.pca_components  = pca_components
        self.bert_batch      = bert_batch
        self.bert_model      = bert_model
        self.bert_fallback   = bert_fallback
        self.seed            = seed

        self._tokenizer  = None
        self._bert_model = None
        self._torch      = None

        self._scaffold_extractor = ScaffoldFeatureExtractor(scaffold_csvs)
        self._desc_imputer = SimpleImputer(strategy="median")
        self._desc_scaler  = StandardScaler()
        self._scaff_scaler = StandardScaler()
        self._pca: Optional[PCA] = None
        self._fitted = False
        self.feature_names: List[str] = []

    # ── BERT ─────────────────────────────────────────────────────────────────

    def _ensure_bert(self):
        if self._tokenizer is None:
            self._tokenizer, self._bert_model_obj, self._torch = _load_chemberta(
                self.bert_model, self.bert_fallback
            )

    def _embed(self, smiles_list: List[str]) -> np.ndarray:
        self._ensure_bert()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._bert_model_obj
        out = []
        for i in range(0, len(smiles_list), self.bert_batch):
            batch = smiles_list[i: i + self.bert_batch]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
            out.append(h)
        return np.vstack(out).astype(np.float32)

    # ── Raw features ─────────────────────────────────────────────────────────

    def _raw(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fps, descs, valid = [], [], []
        for idx, smi in enumerate(smiles_list):
            mol = _parse(smi)
            if mol is None:
                logger.warning(f"Row {idx}: invalid SMILES — excluded.")
                valid.append(False)
                fps.append(np.zeros(self.fp_bits, dtype=np.float32))
                descs.append(np.full(_N_DESC, np.nan, dtype=np.float32))
            else:
                valid.append(True)
                fps.append(_ecfp6(mol, self.fp_bits).astype(np.float32))
                descs.append(_rdkit_desc(mol))
        return np.stack(fps), np.stack(descs), np.array(valid, dtype=bool)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self, smiles_series: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (X_combined, X_tabpfn, X_fp, valid_idx).
        Fits all preprocessors on the training data.
        """
        smiles_list = smiles_series.tolist()
        fps, descs, valid_mask = self._raw(smiles_list)
        valid_idx = np.where(valid_mask)[0]
        n = valid_idx.shape[0]

        fps_v   = fps[valid_mask]
        descs_v = descs[valid_mask]

        # Descriptors
        descs_imp = self._desc_imputer.fit_transform(descs_v)
        descs_sc  = self._desc_scaler.fit_transform(descs_imp)

        # Scaffold features
        valid_smi = smiles_series.iloc[valid_idx].reset_index(drop=True)
        scaff_df  = self._scaffold_extractor.fit_transform(valid_smi)
        scaff_arr = self._scaff_scaler.fit_transform(
            scaff_df.values.astype(np.float32)
        )

        # BERT embeddings
        logger.info(f"Computing ChemBERTa-2 embeddings for {n} molecules ...")
        bert_embs = self._embed(valid_smi.tolist())

        # Combined
        X_combined = np.hstack([bert_embs, descs_sc, fps_v, scaff_arr]).astype(np.float32)

        # PCA for TabPFN
        n_comp = min(self.pca_components, n - 1, X_combined.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=self.seed)
        X_tabpfn = self._pca.fit_transform(X_combined).astype(np.float32)

        # Feature names
        self.feature_names = (
            [f"BERT_{i}" for i in range(bert_embs.shape[1])]
            + list(_DESC_FUNCS.keys())
            + [f"FP_{i}" for i in range(self.fp_bits)]
            + list(scaff_df.columns)
        )

        self._fitted = True
        if (~valid_mask).sum() > 0:
            logger.info(f"Excluded {(~valid_mask).sum()} invalid SMILES.")
        logger.info(
            f"Features — X_combined:{X_combined.shape}  "
            f"X_tabpfn:{X_tabpfn.shape}  X_fp:{fps_v.shape}"
        )
        return X_combined, X_tabpfn, fps_v, valid_idx

    def transform(
        self, smiles_series: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")

        smiles_list = smiles_series.tolist()
        fps, descs, valid_mask = self._raw(smiles_list)
        valid_idx = np.where(valid_mask)[0]

        fps_v   = fps[valid_mask]
        descs_v = descs[valid_mask]

        descs_imp = self._desc_imputer.transform(descs_v)
        descs_sc  = self._desc_scaler.transform(descs_imp)

        valid_smi = smiles_series.iloc[valid_idx].reset_index(drop=True)
        scaff_df  = self._scaffold_extractor.transform(valid_smi)
        scaff_arr = self._scaff_scaler.transform(
            scaff_df.values.astype(np.float32)
        )

        logger.info(f"Computing ChemBERTa-2 embeddings for {len(valid_smi)} molecules ...")
        bert_embs = self._embed(valid_smi.tolist())

        X_combined = np.hstack([bert_embs, descs_sc, fps_v, scaff_arr]).astype(np.float32)
        X_tabpfn   = self._pca.transform(X_combined).astype(np.float32)

        return X_combined, X_tabpfn, fps_v, valid_idx

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        # Exclude heavy BERT weights — re-loaded from HuggingFace cache
        _tok, _mdl, _torch = self._tokenizer, self._bert_model_obj if hasattr(self, '_bert_model_obj') else None, self._torch
        self._tokenizer    = None
        self._bert_model_obj = None
        self._torch        = None
        joblib.dump(self, path)
        self._tokenizer    = _tok
        if _mdl: self._bert_model_obj = _mdl
        self._torch        = _torch
        logger.info(f"Featurizer saved → {path}")

    @classmethod
    def load(cls, path) -> "MolecularFeaturizer":
        feat = joblib.load(path)
        logger.info(f"Featurizer loaded ← {path}")
        return feat
