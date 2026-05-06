from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

# -------------------------------------------------------------------------
# Route-Dependent Decision Policy
# A central design goal of the live workflow is to avoid treating all
# chemistry as equally familiar. These thresholds therefore become more
# conservative as the query moves away from the training domain, reflecting
# the dissertation's applicability-domain and uncertainty rationale.
# -------------------------------------------------------------------------
ROUTE_DECISION_THRESHOLDS = {
    "route_a": 0.50,
    "route_b": 0.55,
    "route_c": 0.65,
}

BASE_ENGINE_WEIGHTS = {
    "descriptor_fp_ensemble": 0.40,
    "full_coverage_ensemble": 0.40,
    "tanimoto_similarity": 0.20,
}


# -------------------------------------------------------------------------
# Applicability-Domain Summary
# The arbitrator receives compact chemistry-space diagnostics rather than raw
# structures. This keeps the consensus layer model-agnostic while still
# exposing the domain variables needed for route selection and confidence
# discounting.
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainMetrics:
    max_tanimoto: float
    mean_tanimoto_top5: float
    n_close_neighbours: int
    scaffold_in_training: bool
    n_alerts_fired: int = 0
    scaffold_toxic_ratio: float | None = None


# -------------------------------------------------------------------------
# Binary Label Extraction
# The codebase stores toxic/non-toxic probabilities as the primary outputs of
# every engine. This helper converts those probabilities into decision labels
# under whichever threshold is appropriate for the current route.
# -------------------------------------------------------------------------
def _label_from_probability(probability: float, threshold: float = 0.5) -> int:
    return int(probability >= threshold)


# -------------------------------------------------------------------------
# Consensus Arbitration
# This class implements the live decision policy described in the hybrid
# cascade: route selection from Tanimoto-domain signals, route-dependent
# engine selection, and consensus generation with transparent disagreement
# handling. The goal is not only predictive performance, but also honest
# deployment behaviour on familiar versus novel chemistry.
# -------------------------------------------------------------------------
class ConsensusArbitrator:
    def __init__(self, route_a_threshold: float = 0.5, route_b_threshold: float = 0.3) -> None:
        self.route_a_threshold = float(route_a_threshold)
        self.route_b_threshold = float(route_b_threshold)

    def choose_route(self, domain: DomainMetrics) -> dict[str, Any]:
        if domain.max_tanimoto >= self.route_a_threshold:
            return {
                "route": "route_a",
                "badge": "In domain",
                "warning": None,
                "reason": "High structural similarity to training chemistry.",
            }
        if domain.max_tanimoto >= self.route_b_threshold:
            return {
                "route": "route_b",
                "badge": "Borderline",
                "warning": None,
                "reason": "Intermediate similarity suggests a transitional applicability zone.",
            }
        return {
            "route": "route_c",
            "badge": "Out of domain",
            "warning": "Novel chemistry detected. Predictions should be treated as advisory and confirmed experimentally.",
            "reason": "Low similarity to the training chemistry space.",
        }

    def _pick_route_predictions(self, route: str, predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        available = [payload for payload in predictions.values() if payload.get("available")]
        if route == "route_a":
            preferred = [
                payload
                for payload in available
                if payload.get("engine_name") == "descriptor_fp_ensemble"
            ]
            return preferred or available[:1]
        if route == "route_b":
            return available
        return available

    def _weighted_consensus_probability(
        self,
        route: str,
        considered: list[dict[str, Any]],
        domain_metrics: DomainMetrics,
    ) -> tuple[float, list[dict[str, Any]]]:
        # When multiple engines are active, their probabilities are not treated
        # equally. The weighting scheme preserves the final benchmark ranking
        # while discounting all engines on distant chemistry and discounting
        # the Tanimoto engine further when its nearest neighbours are weak.
        if len(considered) == 1:
            item = considered[0]
            return float(item["predicted_score"]), [
                {
                    "engine_name": item["engine_name"],
                    "probability": float(item["predicted_score"]),
                    "base_weight": BASE_ENGINE_WEIGHTS.get(item["engine_name"], 1.0),
                    "adjusted_weight": 1.0,
                }
            ]

        similarity_factor = min(domain_metrics.max_tanimoto / 0.5, 1.0)
        tanimoto_discount = max(0.0, min(domain_metrics.mean_tanimoto_top5, 1.0))

        weighted_terms: list[tuple[float, float]] = []
        weight_debug: list[dict[str, Any]] = []
        for item in considered:
            name = str(item["engine_name"])
            probability = float(item["predicted_score"])
            base_weight = BASE_ENGINE_WEIGHTS.get(name, 1.0 / max(len(considered), 1))
            adjusted_weight = base_weight * (tanimoto_discount if name == "tanimoto_similarity" else similarity_factor)
            weighted_terms.append((probability, adjusted_weight))
            weight_debug.append(
                {
                    "engine_name": name,
                    "probability": probability,
                    "base_weight": base_weight,
                    "adjusted_weight": adjusted_weight,
                }
            )

        total_weight = sum(weight for _, weight in weighted_terms)
        if total_weight <= 0:
            fallback_probability = sum(probability for probability, _ in weighted_terms) / len(weighted_terms)
            return float(fallback_probability), weight_debug

        probability = sum(probability * weight for probability, weight in weighted_terms) / total_weight
        return float(probability), weight_debug

    def run(
        self,
        domain_metrics: DomainMetrics,
        predictions: dict[str, dict[str, Any]],
        available_engines: list[str] | None = None,
    ) -> dict[str, Any]:
        route_info = self.choose_route(domain_metrics)
        considered = self._pick_route_predictions(route_info["route"], predictions)

        if not considered:
            return {
                "route": route_info["route"],
                "badge": route_info["badge"],
                "warning": route_info["warning"],
                "reason": route_info["reason"],
                "consensus_label": None,
                "consensus_probability": None,
                "agreement": "no_live_models",
                "engines_considered": [],
                "engine_count": 0,
                "available_engine_names": available_engines or [],
                "domain_metrics": asdict(domain_metrics),
                "message": "No serialized live model artifacts were available, so only the chemistry-native explanation stack is returned.",
            }

        decision_threshold = ROUTE_DECISION_THRESHOLDS[route_info["route"]]
        consensus_probability, weight_debug = self._weighted_consensus_probability(
            route_info["route"],
            considered,
            domain_metrics,
        )
        # This additional guardrail exists for route-C molecules with no
        # named structural alerts. It reduces overconfident toxic calls when
        # the chemistry is distant and the interpretable neighbour evidence is
        # both weak and inconsistent with a strong hazard claim.
        if route_info["route"] == "route_c" and domain_metrics.n_alerts_fired == 0:
            tanimoto_result = predictions.get("tanimoto_similarity", {})
            neighbours = (tanimoto_result.get("details") or {}).get("nearest_neighbours", [])
            if neighbours:
                toxic_ratio = sum(1 for item in neighbours if int(item.get("label", 0)) == 1) / len(neighbours)
                max_similarity = max(float(item.get("similarity", 0.0)) for item in neighbours)
            else:
                toxic_ratio = 0.5
                max_similarity = 0.0
            if max_similarity < 0.20 and toxic_ratio < 0.8:
                consensus_probability *= 0.75

        labels = [_label_from_probability(float(item["predicted_score"]), threshold=decision_threshold) for item in considered]
        label_counter = Counter(labels)
        majority_label, majority_votes = label_counter.most_common(1)[0]
        consensus_label = _label_from_probability(consensus_probability, threshold=decision_threshold)
        unanimous = len(label_counter) == 1

        disagreement = [
            item["engine_name"]
            for item in considered
            if _label_from_probability(float(item["predicted_score"]), threshold=decision_threshold) != majority_label
        ]
        if unanimous:
            agreement = "unanimous"
        elif majority_votes > len(considered) / 2:
            agreement = "majority"
        else:
            agreement = "split"

        return {
            "route": route_info["route"],
            "badge": route_info["badge"],
            "warning": route_info["warning"],
            "reason": route_info["reason"],
            "decision_threshold": float(decision_threshold),
            "consensus_label": int(consensus_label),
            "consensus_probability": float(consensus_probability),
            "agreement": agreement,
            "engines_considered": considered,
            "engine_count": len(considered),
            "dissenting_engines": disagreement,
            "engine_weighting": weight_debug,
            "effective_engine_labels": [
                {
                    "engine_name": item["engine_name"],
                    "effective_label": _label_from_probability(float(item["predicted_score"]), threshold=decision_threshold),
                    "predicted_score": float(item["predicted_score"]),
                }
                for item in considered
            ],
            "available_engine_names": available_engines or [],
            "domain_metrics": asdict(domain_metrics),
            "message": (
                "All route-selected engines agreed after route-dependent thresholding."
                if agreement == "unanimous"
                else "Majority vote returned the final class after route-dependent thresholding; dissenting engines are flagged explicitly."
            ),
        }
