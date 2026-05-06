from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from .descriptors import DescriptorScaler, compute_descriptor_matrix

ATOM_LIST = list(range(1, 119))
DEGREE_LIST = list(range(0, 11))
FORMAL_CHARGE_LIST = list(range(-5, 6))
NUM_H_LIST = list(range(0, 9))
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


@dataclass
class CleanedFrame:
    df: pd.DataFrame
    smiles_col: str
    label_col: str


class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_list: List[Data]):
        super().__init__(None)
        self.data, self.slices = self.collate(data_list)



def set_rdkit_silent() -> None:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")



def one_hot_with_unknown(value, choices: Sequence) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    encoding[idx] = 1
    return encoding



def atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    return (
        one_hot_with_unknown(atom.GetAtomicNum(), ATOM_LIST)
        + one_hot_with_unknown(atom.GetDegree(), DEGREE_LIST)
        + one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGE_LIST)
        + one_hot_with_unknown(atom.GetTotalNumHs(), NUM_H_LIST)
        + one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATION_LIST)
        + [int(atom.GetIsAromatic())]
        + [int(atom.IsInRing())]
        + [atom.GetMass() * 0.01]
        + [int(atom.GetChiralTag())]
    )



def bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    return (
        one_hot_with_unknown(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
        + one_hot_with_unknown(
            bond.GetStereo(),
            [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOANY,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
            ],
        )
    )



def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)



def infer_smiles_column(columns: Sequence[str]) -> str:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in ["smiles", "canonical smiles", "canonical_smiles", "canon_smiles"]:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError("Could not find a SMILES column. Supported names include 'smiles' and 'canonical SMILES'.")



def clean_dataframe(csv_path: str, label_col: str = "label") -> CleanedFrame:
    df = pd.read_csv(csv_path)
    smiles_col = infer_smiles_column(df.columns)
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {csv_path}")

    work = df[[smiles_col, label_col]].copy()
    work = work.dropna(subset=[smiles_col, label_col])
    work[smiles_col] = work[smiles_col].astype(str).str.strip()
    work[label_col] = work[label_col].astype(int)

    if not set(work[label_col].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    work[smiles_col] = work[smiles_col].apply(canonicalize_smiles)
    work = work.dropna(subset=[smiles_col]).drop_duplicates(subset=[smiles_col]).reset_index(drop=True)

    if work.empty:
        raise ValueError(f"No valid molecules found after cleaning {csv_path}")

    return CleanedFrame(df=work, smiles_col=smiles_col, label_col=label_col)



def smiles_to_data(smiles: str, label: Optional[int] = None, descriptors: Optional[list[float]] = None) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_attrs = [], []
    edge_feat_dim = 11
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bf, bf])
        edge_feat_dim = len(bf)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_feat_dim), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if descriptors is not None:
        data.descriptors = torch.tensor(descriptors, dtype=torch.float).view(1, -1)
    if label is not None:
        data.y = torch.tensor([float(label)], dtype=torch.float)
    return data



def dataframe_to_graphs(df: pd.DataFrame, smiles_col: str, label_col: str) -> List[Data]:
    out: List[Data] = []
    for _, row in df.iterrows():
        graph = smiles_to_data(row[smiles_col], int(row[label_col]))
        if graph is not None:
            out.append(graph)
    if not out:
        raise ValueError("No graphs could be created from the dataset.")
    return out



def dataframe_to_hybrid_graphs(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    scaler: DescriptorScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[List[Data], DescriptorScaler]:
    descriptor_matrix = compute_descriptor_matrix(df[smiles_col].tolist())
    if scaler is None:
        scaler = DescriptorScaler()
    if fit_scaler:
        descriptor_matrix = scaler.fit_transform(descriptor_matrix)
    else:
        descriptor_matrix = scaler.transform(descriptor_matrix)

    out: List[Data] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        graph = smiles_to_data(row[smiles_col], int(row[label_col]), descriptors=descriptor_matrix[idx].tolist())
        if graph is not None:
            out.append(graph)
    if not out:
        raise ValueError("No hybrid graphs could be created from the dataset.")
    return out, scaler
