from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neighbors import KNeighborsClassifier


PROTECTED_BASELINE_NAME = "ens_r2_b1024_dice_k31_25_15_isotonic_thr040"
PROTECTED_RADIUS = 2
PROTECTED_BITS = 1024
PROTECTED_METRIC = "dice"
PROTECTED_KS = (31, 25, 15)
PROTECTED_CALIBRATION = "isotonic"
PROTECTED_THRESHOLD = 0.40


@dataclass(frozen=True)
class BaselineConfig:
    name: str = PROTECTED_BASELINE_NAME
    radius: int = PROTECTED_RADIUS
    bits: int = PROTECTED_BITS
    metric: str = PROTECTED_METRIC
    ks: Tuple[int, int, int] = PROTECTED_KS
    calibration: str = PROTECTED_CALIBRATION
    threshold: float = PROTECTED_THRESHOLD


@dataclass
class BaselinePredictionBundle:
    probs: np.ndarray
    component_probs: Dict[str, np.ndarray]
    confidence_abs: np.ndarray
    confidence_margin: np.ndarray
    confidence_vote: np.ndarray


def build_morgan_matrix(smiles: Sequence[str], radius: int, bits: int) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    rows = [np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), dtype=np.uint8) for s in smiles]
    return np.vstack(rows).astype(bool)


def _vote_entropy_confidence(votes_pos: np.ndarray) -> np.ndarray:
    eps = 1e-8
    p = np.clip(votes_pos, eps, 1.0 - eps)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return 1.0 - entropy


def _nearest_class_margin(
    neighbor_distances: np.ndarray,
    neighbor_labels: np.ndarray,
) -> np.ndarray:
    d_pos = np.where(neighbor_labels == 1, neighbor_distances, np.inf).min(axis=1)
    d_neg = np.where(neighbor_labels == 0, neighbor_distances, np.inf).min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.abs(d_neg - d_pos) / (d_neg + d_pos + 1e-8)
    margin = np.nan_to_num(margin, nan=1.0, posinf=1.0, neginf=1.0)
    # If one class never appears in neighbors, treat as maximal margin confidence.
    margin[np.isinf(d_pos) | np.isinf(d_neg)] = 1.0
    return np.clip(margin, 0.0, 1.0)


def _fit_single_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_target: np.ndarray,
    metric: str,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf = KNeighborsClassifier(
        n_neighbors=int(k),
        metric=metric,
        weights="distance",
        algorithm="brute",
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_target)[:, 1]
    distances, indices = clf.kneighbors(x_target, n_neighbors=int(k), return_distance=True)
    neighbor_labels = y_train[indices]
    margin_conf = _nearest_class_margin(distances, neighbor_labels)
    hard_votes = (probs >= 0.5).astype(float)
    return probs, margin_conf, hard_votes


def predict_protected_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_target: np.ndarray,
    config: BaselineConfig | None = None,
) -> BaselinePredictionBundle:
    cfg = config or BaselineConfig()
    component_probs: Dict[str, np.ndarray] = {}
    margins: List[np.ndarray] = []
    hard_votes: List[np.ndarray] = []

    for k in cfg.ks:
        probs, margin_conf, votes = _fit_single_knn(
            x_train=x_train,
            y_train=y_train,
            x_target=x_target,
            metric=cfg.metric,
            k=int(k),
        )
        component_probs[f"{cfg.metric}_k{k}"] = probs
        margins.append(margin_conf)
        hard_votes.append(votes)

    component_stack = np.vstack([component_probs[f"{cfg.metric}_k{k}"] for k in cfg.ks])
    probs_ens = component_stack.mean(axis=0)

    conf_abs = np.abs(probs_ens - 0.5)
    conf_margin = np.mean(np.vstack(margins), axis=0)
    vote_ratio = np.mean(np.vstack(hard_votes), axis=0)
    conf_vote = _vote_entropy_confidence(vote_ratio)

    return BaselinePredictionBundle(
        probs=probs_ens,
        component_probs=component_probs,
        confidence_abs=conf_abs,
        confidence_margin=conf_margin,
        confidence_vote=conf_vote,
    )
