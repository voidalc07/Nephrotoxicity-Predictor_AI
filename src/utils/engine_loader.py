from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config.settings import LIVE_ASSET_FINGERPRINTS, MODEL_ARTIFACTS_DIR


# -------------------------------------------------------------------------
# Engine Status and Prediction Records
# Live inference may run in a lightweight portable build where some model
# artefacts are intentionally absent. These dataclasses make capability
# reporting explicit so the dashboard can distinguish "engine unavailable"
# from an actual prediction result.
# -------------------------------------------------------------------------
@dataclass
class EngineStatus:
    engine_name: str
    display_name: str
    available: bool
    artifact_path: str
    reason_unavailable: str | None = None
    notes: str | None = None


@dataclass
class EnginePrediction:
    engine_name: str
    display_name: str
    available: bool
    predicted_score: float | None = None
    predicted_label: int | None = None
    threshold: float | None = None
    reason_unavailable: str | None = None
    details: dict[str, Any] | None = None


# -------------------------------------------------------------------------
# Base Runtime Engine Interface
# Each live engine must support two operations: artefact loading at server
# startup and per-query probability prediction at request time. This small
# abstraction allows similarity-based and serialized ensemble engines to be
# exposed through one registry.
# -------------------------------------------------------------------------
class BaseEngine:
    def __init__(self, engine_name: str, display_name: str, artifact_path: Path, notes: str | None = None) -> None:
        self.engine_name = engine_name
        self.display_name = display_name
        self.artifact_path = artifact_path
        self.notes = notes
        self.available = False
        self.reason_unavailable: str | None = None

    def status(self) -> EngineStatus:
        return EngineStatus(
            engine_name=self.engine_name,
            display_name=self.display_name,
            available=self.available,
            artifact_path=str(self.artifact_path),
            reason_unavailable=self.reason_unavailable,
            notes=self.notes,
        )

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, features: dict[str, Any]) -> EnginePrediction:
        raise NotImplementedError


# -------------------------------------------------------------------------
# Portable Tanimoto Similarity Engine
# This engine is the always-on interpretability backbone of the live system.
# It performs similarity-weighted read-across against the bundled training
# fingerprint matrix, providing an applicability-domain-friendly fallback
# when heavier serialized models are not present.
# -------------------------------------------------------------------------
class TanimotoSimilarityEngine(BaseEngine):
    def __init__(self, asset_path: Path = LIVE_ASSET_FINGERPRINTS) -> None:
        super().__init__(
            engine_name="tanimoto_similarity",
            display_name="Tanimoto Similarity Engine",
            artifact_path=asset_path,
            notes="Portable fallback using the bundled training fingerprint matrix.",
        )
        self.top_k = 5
        self.training_fingerprints: np.ndarray | None = None
        self.training_labels: np.ndarray | None = None
        self.training_smiles: np.ndarray | None = None

    def load(self) -> None:
        if not self.artifact_path.exists():
            self.available = False
            self.reason_unavailable = f"Training fingerprint context not found at {self.artifact_path}."
            return
        with np.load(self.artifact_path, allow_pickle=False) as loaded:
            self.training_fingerprints = loaded["fingerprints"].astype(np.uint8)
            self.training_labels = loaded["labels"].astype(np.int8)
            self.training_smiles = loaded["smiles"].astype(str)
        self.available = True
        self.reason_unavailable = None

    def predict(self, features: dict[str, Any]) -> EnginePrediction:
        if not self.available or self.training_fingerprints is None or self.training_labels is None or self.training_smiles is None:
            return EnginePrediction(
                engine_name=self.engine_name,
                display_name=self.display_name,
                available=False,
                reason_unavailable=self.reason_unavailable or "Training fingerprint context is unavailable.",
            )

        fingerprint = np.asarray(features.get("morgan_fp"), dtype=np.uint8)
        if fingerprint.size == 0:
            return EnginePrediction(
                engine_name=self.engine_name,
                display_name=self.display_name,
                available=False,
                reason_unavailable="The live feature payload did not include a Morgan fingerprint array.",
            )

        # The probability is computed by similarity-weighted neighbour voting.
        # This is intentionally simple and transparent: unlike a black-box
        # classifier, the returned score can be traced to explicit structural
        # analogues in the training corpus.
        intersections = self.training_fingerprints @ fingerprint
        query_on = int(fingerprint.sum())
        train_on = self.training_fingerprints.sum(axis=1)
        unions = train_on + query_on - intersections
        similarities = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)
        top_indices = np.argsort(similarities)[::-1][: self.top_k]
        top_scores = similarities[top_indices]
        weights = top_scores + 1e-8
        weighted_probability = float(np.sum(weights * self.training_labels[top_indices]) / np.sum(weights))
        neighbours = [
            {
                "rank": rank + 1,
                "canonical_smiles": str(self.training_smiles[index]),
                "label": int(self.training_labels[index]),
                "similarity": float(similarities[index]),
            }
            for rank, index in enumerate(top_indices)
        ]
        return EnginePrediction(
            engine_name=self.engine_name,
            display_name=self.display_name,
            available=True,
            predicted_score=weighted_probability,
            predicted_label=int(weighted_probability >= 0.5),
            threshold=0.5,
            details={"nearest_neighbours": neighbours, "top_similarity": float(top_scores[0]) if len(top_scores) else None},
        )


# -------------------------------------------------------------------------
# Serialized Stacking Ensemble Engine
# The descriptor/fingerprint and full-coverage live models share the same
# runtime structure: imputation, scaling, two base learners, and a logistic
# meta-learner trained on their out-of-fold probabilities. The engine manifest
# preserves feature order so live featurisation matches training exactly.
# -------------------------------------------------------------------------
class StackingEnsembleEngine(BaseEngine):
    def __init__(self, engine_name: str, display_name: str, engine_dir: Path, notes: str | None = None) -> None:
        super().__init__(engine_name=engine_name, display_name=display_name, artifact_path=engine_dir, notes=notes)
        self.metadata: dict[str, Any] = {}
        self.models: dict[str, Any] = {}

    def load(self) -> None:
        manifest_path = self.artifact_path / "engine.json"
        if not manifest_path.exists():
            self.available = False
            self.reason_unavailable = f"Engine manifest not found at {manifest_path}."
            return

        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            required = metadata.get("artifacts", {})
            for key, relative_path in required.items():
                self.models[key] = joblib.load(self.artifact_path / relative_path)
            self.metadata = metadata
        except FileNotFoundError as exc:
            self.available = False
            self.reason_unavailable = f"Missing serialized artifact: {exc}"
            return
        except Exception as exc:  # pragma: no cover
            self.available = False
            self.reason_unavailable = f"Failed to load serialized artifacts: {exc}"
            return

        self.available = True
        self.reason_unavailable = None

    def _vector_from_features(self, features: dict[str, Any], feature_order: list[str]) -> np.ndarray:
        values = [float(features.get(name, 0.0)) for name in feature_order]
        return np.asarray(values, dtype=float).reshape(1, -1)

    def predict(self, features: dict[str, Any]) -> EnginePrediction:
        if not self.available:
            return EnginePrediction(
                engine_name=self.engine_name,
                display_name=self.display_name,
                available=False,
                reason_unavailable=self.reason_unavailable or "Serialized artifacts are unavailable.",
            )

        feature_order = list(self.metadata.get("feature_order", []))
        if not feature_order:
            return EnginePrediction(
                engine_name=self.engine_name,
                display_name=self.display_name,
                available=False,
                reason_unavailable="No feature order was defined in the engine manifest.",
            )

        # Ordered feature reconstruction is critical here. Tree and k-NN models
        # trained on descriptor/fingerprint matrices are highly sensitive to
        # column ordering, so the manifest acts as a reproducibility contract.
        x = self._vector_from_features(features, feature_order)
        imputer = self.models.get("imputer")
        scaler = self.models.get("scaler")
        x_base = imputer.transform(x) if imputer is not None else x
        x_scaled = scaler.transform(x_base) if scaler is not None else x_base

        component_scores: dict[str, float] = {}
        meta_feature_order = list(self.metadata.get("meta_feature_order", []))
        # The base learners generate complementary views of the same molecule:
        # a local-neighbour estimate from k-NN and a nonlinear tabular signal
        # from LightGBM over sparse cheminformatics features.
        for component_name in meta_feature_order:
            model = self.models.get(component_name)
            if model is None:
                continue
            model_input: Any = x_scaled
            if hasattr(model, "feature_names_in_"):
                model_input = pd.DataFrame(x_scaled, columns=feature_order)
            component_scores[component_name] = float(model.predict_proba(model_input)[:, 1][0])

        if not component_scores:
            return EnginePrediction(
                engine_name=self.engine_name,
                display_name=self.display_name,
                available=False,
                reason_unavailable="No component models were available for this engine.",
            )

        meta_model = self.models.get("meta_model")
        calibrator = self.models.get("calibrator")
        threshold = float(self.metadata.get("threshold", 0.5))
        # The meta-learner implements classic stacked ensembling, using the
        # calibrated or raw base probabilities as low-dimensional decision
        # features rather than re-consuming the full molecular descriptor set.
        if meta_model is not None:
            meta_values = np.asarray([component_scores[name] for name in meta_feature_order if name in component_scores], dtype=float).reshape(1, -1)
            raw_score = float(meta_model.predict_proba(meta_values)[:, 1][0])
            if calibrator is not None:
                score = float(calibrator.predict_proba(np.asarray([[raw_score]], dtype=float))[:, 1][0])
            else:
                score = raw_score
        else:
            score = float(sum(component_scores.values()) / len(component_scores))

        return EnginePrediction(
            engine_name=self.engine_name,
            display_name=self.display_name,
            available=True,
            predicted_score=score,
            predicted_label=int(score >= threshold),
            threshold=threshold,
            details={
                "component_scores": component_scores,
                "calibration": self.metadata.get("calibration"),
            },
        )


# -------------------------------------------------------------------------
# Engine Registry
# The registry is the runtime discovery layer for the live dashboard. It
# instantiates all supported engines, loads what is available in the current
# deployment profile, and exposes a uniform prediction interface to the
# consensus and explanation stack.
# -------------------------------------------------------------------------
class EngineRegistry:
    def __init__(self, models_root: str | Path = MODEL_ARTIFACTS_DIR) -> None:
        self.models_root = Path(models_root)
        self.engines: dict[str, BaseEngine] = {
            "descriptor_fp_ensemble": StackingEnsembleEngine(
                "descriptor_fp_ensemble",
                "Descriptor + Fingerprint Ensemble",
                self.models_root / "descriptor_fp_ensemble",
                notes="Loads imputer, scaler, base models, and logistic meta-learner when serialized artifacts exist.",
            ),
            "full_coverage_ensemble": StackingEnsembleEngine(
                "full_coverage_ensemble",
                "Full Coverage Ensemble",
                self.models_root / "full_coverage_ensemble",
                notes="Same stacking architecture as the descriptor engine with a broader feature set.",
            ),
            "tanimoto_similarity": TanimotoSimilarityEngine(),
        }

    def load(self) -> None:
        for engine in self.engines.values():
            engine.load()

    @property
    def available_engines(self) -> list[str]:
        return [name for name, engine in self.engines.items() if engine.available]

    def get_statuses(self) -> list[dict[str, Any]]:
        return [asdict(engine.status()) for engine in self.engines.values()]

    def predict(self, engine_name: str, features: dict[str, Any]) -> dict[str, Any]:
        if engine_name not in self.engines:
            return asdict(
                EnginePrediction(
                    engine_name=engine_name,
                    display_name=engine_name,
                    available=False,
                    reason_unavailable="Engine is not registered.",
                )
            )
        return asdict(self.engines[engine_name].predict(features))

    def predict_available(self, features: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {name: asdict(engine.predict(features)) for name, engine in self.engines.items()}
