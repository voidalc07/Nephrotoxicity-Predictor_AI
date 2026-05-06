from __future__ import annotations

import csv
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import DEFAULT_TRAIN_CSV, DEPLOYMENT_MODE, FEEDBACK_CSV, LIVE_ASSET_FINGERPRINTS
from src.utils.chemistry_alerts import ALERT_DEFINITIONS
from src.utils.consensus import ConsensusArbitrator, DomainMetrics
from src.utils.engine_loader import EngineRegistry
from src.utils.io import safe_read_csv

try:
    from rdkit import Chem, DataStructs
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None
    Descriptors = None
    MurckoScaffold = None


# -------------------------------------------------------------------------
# Live Analysis Configuration
# The live workflow mirrors the dissertation's hybrid cascade while remaining
# lightweight enough for portable deployment. Route thresholds are expressed
# in Tanimoto space because structural similarity is the most interpretable
# applicability-domain signal available across all deployment profiles.
# -------------------------------------------------------------------------
TOP_NEIGHBOURS = 5
FINGERPRINT_BITS = 2048
ROUTE_A_SIMILARITY = 0.5
ROUTE_B_SIMILARITY = 0.3

_CONTEXT_CACHE: dict[str, Any] = {"context": None, "mtime": None}
_ALERT_PATTERNS: list[tuple[dict[str, str], Any]] | None = None
_REGISTRY: EngineRegistry | None = None
_ARBITRATOR = ConsensusArbitrator(route_a_threshold=ROUTE_A_SIMILARITY, route_b_threshold=ROUTE_B_SIMILARITY)


# -------------------------------------------------------------------------
# RDKit Availability and Input Normalisation
# RDKit underpins canonicalisation, fingerprints, descriptors, SMARTS, and
# scaffold extraction. These helpers concentrate the dependency boundary so
# live prediction fails early and clearly if the cheminformatics layer is
# unavailable or the query SMILES cannot be parsed.
# -------------------------------------------------------------------------
def _rdkit_ready() -> bool:
    return Chem is not None and AllChem is not None and DataStructs is not None and Descriptors is not None and MurckoScaffold is not None


def _pick_smiles_column(frame: pd.DataFrame) -> str:
    normalized = {col.strip().lower().replace("_", " "): col for col in frame.columns}
    for key in ("canonical smiles", "canonical_smiles", "smiles", "smiles clean"):
        if key in normalized:
            return normalized[key]
    raise ValueError("Could not find a SMILES column in the training dataset.")


def _mol_from_smiles(smiles: str):
    if not _rdkit_ready():
        raise RuntimeError("RDKit is required for live analysis but is not available in this environment.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("The supplied SMILES string could not be parsed.")
    return mol


def _bitvect_to_array(bitvect: Any) -> np.ndarray:
    array = np.zeros((FINGERPRINT_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array


def _canonicalize_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(_mol_from_smiles(smiles))


def _murcko_scaffold_smiles(mol: Any) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


# -------------------------------------------------------------------------
# SMARTS Structural Alert Compilation
# Alerts are compiled once and cached so live requests can expose named
# toxicophore-style explanations without repeatedly rebuilding SMARTS queries.
# This is a lightweight analogue of structure-alert systems often used in
# safety screening to complement purely statistical classifiers.
# -------------------------------------------------------------------------
def _alert_patterns() -> list[tuple[dict[str, str], Any]]:
    global _ALERT_PATTERNS
    if _ALERT_PATTERNS is None:
        compiled: list[tuple[dict[str, str], Any]] = []
        for definition in ALERT_DEFINITIONS:
            compiled.append((definition, Chem.MolFromSmarts(definition["smarts"]) if Chem is not None else None))
        _ALERT_PATTERNS = compiled
    return _ALERT_PATTERNS


def _query_alerts(mol: Any) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for definition, pattern in _alert_patterns():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            matches.append(
                {
                    "name": definition["name"],
                    "smarts": definition["smarts"],
                    "reference": definition["reference"],
                }
            )
    return matches


def _descriptor_features(mol: Any) -> dict[str, float]:
    return {name: float(fn(mol)) for name, fn in Descriptors._descList}


# -------------------------------------------------------------------------
# Training-Context Construction
# The applicability-domain gate and nearest-neighbour explanation layer rely
# on a precomputed training fingerprint matrix. Persisting this matrix to a
# compact `.npz` file makes live Tanimoto lookup feasible without loading the
# full historical training pipeline at dashboard startup.
# -------------------------------------------------------------------------
def _build_training_context() -> dict[str, Any]:
    train_df = safe_read_csv(DEFAULT_TRAIN_CSV)
    smiles_col = _pick_smiles_column(train_df)

    smiles_values: list[str] = []
    labels: list[int] = []
    scaffolds: list[str] = []
    fingerprints: list[np.ndarray] = []

    for _, row in train_df.iterrows():
        smiles = str(row[smiles_col]).strip()
        if not smiles:
            continue
        try:
            mol = _mol_from_smiles(smiles)
        except ValueError:
            continue
        smiles_values.append(Chem.MolToSmiles(mol))
        labels.append(int(row.get("label", 0)))
        scaffolds.append(_murcko_scaffold_smiles(mol))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FINGERPRINT_BITS)
        fingerprints.append(_bitvect_to_array(fp))

    if not fingerprints:
        raise RuntimeError("Could not build a training fingerprint context from the model construction dataset.")

    matrix = np.vstack(fingerprints).astype(np.uint8)
    LIVE_ASSET_FINGERPRINTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LIVE_ASSET_FINGERPRINTS,
        fingerprints=matrix,
        smiles=np.array(smiles_values, dtype=str),
        labels=np.array(labels, dtype=np.int8),
        scaffolds=np.array(scaffolds, dtype=str),
        generated_at=np.array([datetime.now(UTC).isoformat()], dtype=str),
    )
    return {
        "fingerprints": matrix,
        "smiles": np.array(smiles_values, dtype=str),
        "labels": np.array(labels, dtype=np.int8),
        "scaffolds": np.array(scaffolds, dtype=str),
        "generated_at": datetime.now(UTC).isoformat(),
        "asset_path": str(LIVE_ASSET_FINGERPRINTS),
        "rebuilt": True,
    }


def load_training_context(force_rebuild: bool = False) -> dict[str, Any]:
    current_mtime = DEFAULT_TRAIN_CSV.stat().st_mtime if DEFAULT_TRAIN_CSV.exists() else None
    if (
        not force_rebuild
        and _CONTEXT_CACHE["context"] is not None
        and _CONTEXT_CACHE["mtime"] == current_mtime
    ):
        return _CONTEXT_CACHE["context"]

    if force_rebuild or not LIVE_ASSET_FINGERPRINTS.exists():
        context = _build_training_context()
    else:
        with np.load(LIVE_ASSET_FINGERPRINTS, allow_pickle=False) as loaded:
            context = {
                "fingerprints": loaded["fingerprints"].astype(np.uint8),
                "smiles": loaded["smiles"].astype(str),
                "labels": loaded["labels"].astype(np.int8),
                "scaffolds": loaded["scaffolds"].astype(str),
                "generated_at": loaded["generated_at"][0].item(),
                "asset_path": str(LIVE_ASSET_FINGERPRINTS),
                "rebuilt": False,
            }
    _CONTEXT_CACHE["context"] = context
    _CONTEXT_CACHE["mtime"] = current_mtime
    return context


# -------------------------------------------------------------------------
# Live Engine Registry Access
# Runtime model loading is cached at module scope because the serialized
# ensemble bundles are heavier than the per-request chemistry transforms.
# This avoids repeated disk reads while keeping the public API simple.
# -------------------------------------------------------------------------
def get_registry() -> EngineRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = EngineRegistry()
        _REGISTRY.load()
    return _REGISTRY


# -------------------------------------------------------------------------
# Query Featurisation
# Every live request is converted into the same chemically meaningful views
# used throughout the project: canonical SMILES, Morgan fingerprints, RDKit
# descriptors, Bemis-Murcko scaffolds, and named SMARTS alert flags.
# -------------------------------------------------------------------------
def build_query_context(smiles: str) -> dict[str, Any]:
    clean = smiles.strip()
    if not clean:
        raise ValueError("Please enter a SMILES string for live prediction.")
    mol = _mol_from_smiles(clean)
    canonical = Chem.MolToSmiles(mol)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FINGERPRINT_BITS)
    fingerprint_array = _bitvect_to_array(fingerprint)
    alerts = _query_alerts(mol)
    return {
        "input_smiles": clean,
        "canonical_smiles": canonical,
        "mol": mol,
        "fingerprint": fingerprint_array,
        "scaffold": _murcko_scaffold_smiles(mol),
        "alerts": alerts,
        "alert_flags": {f"alert_{item['name'].lower().replace(' ', '_')}": 1 for item in alerts},
        "descriptors": _descriptor_features(mol),
    }


# -------------------------------------------------------------------------
# Tanimoto Applicability-Domain Computation
# Maximum and top-k Tanimoto similarity provide a read-across style estimate
# of whether the query resembles training chemistry. This is the primary
# domain signal because it is chemically interpretable and available even in
# minimal portable deployments where probabilistic uncertainty models may be
# absent.
# -------------------------------------------------------------------------
def _tanimoto_against_matrix(query_fp: np.ndarray, training_fps: np.ndarray) -> np.ndarray:
    intersections = training_fps @ query_fp
    query_on = int(query_fp.sum())
    train_on = training_fps.sum(axis=1)
    unions = train_on + query_on - intersections
    return np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)


def _scaffold_context(query_scaffold: str, training_context: dict[str, Any]) -> dict[str, Any]:
    if not query_scaffold:
        return {
            "query_scaffold": "",
            "status": "no_murcko_scaffold",
            "family_size": 0,
            "toxic_count": 0,
            "non_toxic_count": 0,
            "toxic_ratio": None,
            "scaffold_in_training": False,
        }

    scaffolds = training_context["scaffolds"]
    labels = training_context["labels"]
    mask = scaffolds == query_scaffold
    family_size = int(mask.sum())
    if family_size == 0:
        return {
            "query_scaffold": query_scaffold,
            "status": "novel_scaffold",
            "family_size": 0,
            "toxic_count": 0,
            "non_toxic_count": 0,
            "toxic_ratio": None,
            "scaffold_in_training": False,
        }
    toxic_count = int(labels[mask].sum())
    non_toxic_count = int(family_size - toxic_count)
    return {
        "query_scaffold": query_scaffold,
        "status": "matched_training_family",
        "family_size": family_size,
        "toxic_count": toxic_count,
        "non_toxic_count": non_toxic_count,
        "toxic_ratio": float(toxic_count / family_size) if family_size else None,
        "scaffold_in_training": True,
    }


def compute_domain_gate(query_context: dict[str, Any], training_context: dict[str, Any]) -> dict[str, Any]:
    similarities = _tanimoto_against_matrix(query_context["fingerprint"].astype(np.uint8), training_context["fingerprints"].astype(np.uint8))
    top_indices = np.argsort(similarities)[::-1][:TOP_NEIGHBOURS]
    neighbours = [
        {
            "rank": rank + 1,
            "canonical_smiles": str(training_context["smiles"][index]),
            "label": int(training_context["labels"][index]),
            "similarity": float(similarities[index]),
        }
        for rank, index in enumerate(top_indices)
    ]
    max_similarity = float(similarities[top_indices[0]]) if len(top_indices) else 0.0
    mean_top_similarity = float(np.mean(similarities[top_indices])) if len(top_indices) else 0.0
    n_close_neighbours = int(np.sum(similarities >= ROUTE_A_SIMILARITY))
    # Route assignment implements the dissertation's confidence-routed
    # cascade: familiar chemistry enters a high-confidence path, while
    # distant chemistry is explicitly marked as borderline or out-of-domain.
    if max_similarity >= ROUTE_A_SIMILARITY:
        route = "route_a"
        badge = "In domain"
    elif max_similarity >= ROUTE_B_SIMILARITY:
        route = "route_b"
        badge = "Borderline"
    else:
        route = "route_c"
        badge = "Out of domain"
    return {
        "route": route,
        "badge": badge,
        "max_tanimoto": max_similarity,
        "mean_tanimoto_top5": mean_top_similarity,
        "n_close_neighbours": n_close_neighbours,
        "nearest_neighbours": neighbours,
    }


# -------------------------------------------------------------------------
# Feature Payload for Runtime Engines
# The live engines consume a flattened feature dictionary so serialized
# tabular models and the similarity engine can share one interface despite
# relying on different subsets of the molecular representation.
# -------------------------------------------------------------------------
def _build_engine_features(context: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    features.update(context["descriptors"])
    features.update({f"fp_{index}": int(bit) for index, bit in enumerate(context["fingerprint"])})
    features.update(context["alert_flags"])
    features["morgan_fp"] = context["fingerprint"].astype(np.uint8)
    return features


# -------------------------------------------------------------------------
# Layered Explanation Stack
# This object mirrors the dashboard's interpretation panels: neighbour-based
# read-across evidence, SMARTS structural alerts, scaffold-family context,
# and a placeholder for local feature attributions once full serialized
# attribution support is available.
#
# NOTE:
# The current portable build does not yet compute SHAP or equivalent local
# attributions for the live ensemble scores; the feature-importance panel is
# therefore an explicit placeholder rather than a completed explanation layer.
# -------------------------------------------------------------------------
def build_explanation_stack(query_context: dict[str, Any], domain: dict[str, Any], training_context: dict[str, Any]) -> dict[str, Any]:
    scaffold = _scaffold_context(query_context["scaffold"], training_context)
    return {
        "nearest_neighbours": domain["nearest_neighbours"],
        "structural_alerts": query_context["alerts"],
        "scaffold_context": scaffold,
        "feature_importance": {
            "status": "pending_serialized_model_support",
            "message": "Local feature attributions will appear when the serialized ensemble models are available.",
        },
    }


# -------------------------------------------------------------------------
# End-to-End Live Prediction Assembly
# This function is the central live-analysis entry point. It combines the
# chemistry-native domain gate, serialized-engine execution, and consensus
# arbitration into a single payload returned to the dashboard.
# -------------------------------------------------------------------------
def build_live_payload(smiles: str) -> dict[str, Any]:
    training_context = load_training_context()
    query_context = build_query_context(smiles)
    domain = compute_domain_gate(query_context, training_context)
    explanations = build_explanation_stack(query_context, domain, training_context)

    features = _build_engine_features(query_context)
    registry = get_registry()
    predictions = registry.predict_available(features)
    domain_metrics = DomainMetrics(
        max_tanimoto=domain["max_tanimoto"],
        mean_tanimoto_top5=domain["mean_tanimoto_top5"],
        n_close_neighbours=domain["n_close_neighbours"],
        scaffold_in_training=bool(explanations["scaffold_context"]["scaffold_in_training"]),
        n_alerts_fired=len(explanations["structural_alerts"]),
        scaffold_toxic_ratio=explanations["scaffold_context"]["toxic_ratio"],
    )
    consensus = _ARBITRATOR.run(domain_metrics, predictions, registry.available_engines)

    return {
        "query_smiles": query_context["input_smiles"],
        "canonical_smiles": query_context["canonical_smiles"],
        "deployment_mode": DEPLOYMENT_MODE,
        "asset_context": {
            "training_fingerprint_asset": training_context["asset_path"],
            "generated_at": training_context["generated_at"],
            "rebuilt_this_session": bool(training_context.get("rebuilt", False)),
            "training_molecule_count": int(len(training_context["smiles"])),
        },
        "domain": domain,
        "consensus": consensus,
        "engine_predictions": predictions,
        "engine_statuses": registry.get_statuses(),
        "explanations": explanations,
    }


def predict_live(smiles: str) -> dict[str, Any]:
    return build_live_payload(smiles)


# -------------------------------------------------------------------------
# Human-in-the-Loop Feedback Logging
# Confirmed labels are appended to a local CSV rather than directly updating
# the training set. This preserves a clean separation between inference-time
# usage and future retraining, matching the dissertation's lightweight
# feedback-loop design.
# -------------------------------------------------------------------------
def append_feedback(smiles: str, label: int, source: str | None = None, note: str | None = None) -> dict[str, Any]:
    canonical = _canonicalize_smiles(smiles)
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_CSV.exists()
    with FEEDBACK_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "canonical_smiles", "confirmed_label", "source", "note"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "canonical_smiles": canonical,
                "confirmed_label": int(label),
                "source": (source or "").strip(),
                "note": (note or "").strip(),
            }
        )
    return {
        "status": "recorded",
        "canonical_smiles": canonical,
        "confirmed_label": int(label),
        "feedback_path": str(FEEDBACK_CSV),
    }
