from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold



def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)



def carbon_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic, canonical=True) if generic is not None else ""



def get_scaffold(smiles: str, scaffold_type: str = "murcko") -> str:
    scaffold_type = scaffold_type.lower().strip()
    if scaffold_type == "murcko":
        return murcko_scaffold(smiles)
    if scaffold_type == "carbon":
        return carbon_scaffold(smiles)
    raise ValueError("scaffold_type must be 'murcko' or 'carbon'")



def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    val_fraction: float = 0.2,
    scaffold_type: str = "murcko",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    work = df.copy().reset_index(drop=True)
    work["_scaffold"] = work[smiles_col].apply(lambda s: get_scaffold(s, scaffold_type=scaffold_type))

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, scaffold in enumerate(work["_scaffold"].tolist()):
        groups[scaffold].append(idx)

    group_items = sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    target_val = max(1, int(round(len(work) * val_fraction)))

    train_idx: List[int] = []
    val_idx: List[int] = []
    val_pos = 0
    val_neg = 0
    total_pos = int(work[label_col].sum())
    total_neg = len(work) - total_pos
    target_val_pos = int(round(total_pos * val_fraction))
    target_val_neg = int(round(total_neg * val_fraction))

    for _, indices in group_items:
        group = work.iloc[indices]
        group_pos = int(group[label_col].sum())
        group_neg = len(group) - group_pos

        choose_val = False
        if len(val_idx) < target_val:
            future_val_size = len(val_idx) + len(indices)
            current_gap = abs(val_pos - target_val_pos) + abs(val_neg - target_val_neg)
            future_gap = abs((val_pos + group_pos) - target_val_pos) + abs((val_neg + group_neg) - target_val_neg)
            choose_val = future_gap <= current_gap or future_val_size <= target_val

        if choose_val:
            val_idx.extend(indices)
            val_pos += group_pos
            val_neg += group_neg
        else:
            train_idx.extend(indices)

    if not train_idx or not val_idx:
        raise ValueError("Scaffold split failed to create non-empty train and validation sets.")

    train_df = work.iloc[train_idx].drop(columns=["_scaffold"]).reset_index(drop=True)
    val_df = work.iloc[val_idx].drop(columns=["_scaffold"]).reset_index(drop=True)
    return train_df, val_df
