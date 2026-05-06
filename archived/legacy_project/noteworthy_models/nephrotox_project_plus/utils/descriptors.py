from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors


DESCRIPTOR_NAMES = [
    "mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rot_bonds",
    "ring_count",
    "aromatic_rings",
    "fraction_csp3",
    "heavy_atoms",
    "formal_charge",
    "hetero_atoms",
]


class DescriptorScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "DescriptorScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("DescriptorScaler must be fit before transform.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "names": DESCRIPTOR_NAMES,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DescriptorScaler":
        scaler = cls()
        scaler.mean_ = np.array(state["mean"], dtype=np.float32)
        scaler.std_ = np.array(state["std"], dtype=np.float32)
        return scaler



def compute_descriptor_vector(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for descriptors: {smiles}")

    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    vec = np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.FractionCSP3(mol),
            mol.GetNumHeavyAtoms(),
            formal_charge,
            Lipinski.NumHeteroatoms(mol),
        ],
        dtype=np.float32,
    )
    return vec



def compute_descriptor_matrix(smiles_list: Iterable[str]) -> np.ndarray:
    return np.vstack([compute_descriptor_vector(s) for s in smiles_list]).astype(np.float32)
