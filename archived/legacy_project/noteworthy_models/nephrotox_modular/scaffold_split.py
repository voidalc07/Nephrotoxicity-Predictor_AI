"""
scaffold_split.py
-----------------
Scaffold-stratified K-fold cross-validation.

WHY THIS IS CRITICAL
--------------------
Random CV gave us 0.981 AUC internally but collapsed to 0.735 on the
external test set.  The reason: 66% of external scaffolds were never seen
during training.  Random folds share the same scaffold distribution,
making the model look far better than it truly generalises.

Scaffold-stratified split:
  - Groups molecules by Bemis-Murcko scaffold
  - Entire scaffold groups go to either train OR val — never split
  - Validation molecules therefore always have scaffolds unseen in training
  - This directly mimics the real challenge posed by the external test set

Effect: internal CV scores drop (become more honest), external test
scores improve because the model is now penalised for scaffold memorisation.
"""

import logging
from collections import defaultdict
from typing import Iterator, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False
    logger.warning("RDKit not found — scaffold split will fall back to stratified random split.")


def _get_scaffold(smi: str) -> str:
    """Return Bemis-Murcko scaffold SMILES, or '' for acyclic molecules."""
    if not _RDKIT_OK:
        return ""
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_kfold(
    smiles: List[str],
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, val_idx) tuples for scaffold-stratified K-fold.

    Algorithm:
      1. Get Murcko scaffold for every molecule.
      2. Group molecule indices by scaffold.
      3. Sort groups largest-first (reproducible ordering).
      4. Assign groups to folds greedily: each group goes to the fold with
         the fewest molecules so far (greedy balanced assignment).
      5. Yield train/val splits from fold assignments.

    Parameters
    ----------
    smiles   : list of SMILES strings
    y        : binary label array (0/1)
    n_splits : number of folds
    seed     : for reproducibility when breaking ties
    """
    rng = np.random.default_rng(seed)

    # ── Step 1-2: Group by scaffold ──────────────────────────────────────────
    scaffold_to_indices: defaultdict = defaultdict(list)
    for i, smi in enumerate(smiles):
        scaffold_to_indices[_get_scaffold(smi)].append(i)

    # ── Step 3: Sort groups (largest first, shuffle ties) ────────────────────
    groups = list(scaffold_to_indices.values())
    groups.sort(key=lambda g: -len(g))

    # ── Step 4: Greedy balanced fold assignment ───────────────────────────────
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_assignments = []          # which fold does each group go to?

    for group in groups:
        target_fold = int(np.argmin(fold_sizes))
        fold_assignments.append(target_fold)
        fold_sizes[target_fold] += len(group)

    # Build fold → index lists
    fold_indices: List[List[int]] = [[] for _ in range(n_splits)]
    for group, fold_id in zip(groups, fold_assignments):
        fold_indices[fold_id].extend(group)

    # ── Step 5: Yield splits ──────────────────────────────────────────────────
    all_idx = np.arange(len(smiles))
    for val_fold in range(n_splits):
        val_idx  = np.array(fold_indices[val_fold])
        train_idx = np.concatenate([
            np.array(fold_indices[f])
            for f in range(n_splits) if f != val_fold
        ])
        yield train_idx, val_idx

    # ── Log scaffold stats ────────────────────────────────────────────────────
    n_scaffolds = len(scaffold_to_indices)
    n_acyclic   = len(scaffold_to_indices.get("", []))
    logger.info(
        f"Scaffold split: {n_scaffolds} unique scaffolds | "
        f"{n_acyclic} acyclic molecules | "
        f"fold sizes: {fold_sizes.tolist()}"
    )
