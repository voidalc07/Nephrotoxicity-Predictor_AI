from __future__ import annotations

from collections import OrderedDict

from src.models.common import ModelResult, combine_notes
from src.models.legacy.nephro_final import run_registered as run_noteworthy_final
from src.models.legacy.nephro_fixed import run_registered as run_noteworthy_fixed
from src.models.legacy.nephro_modular import (
    run_modular_gin_virtual,
    run_modular_histgb,
    run_modular_lightgbm,
    run_modular_node,
    run_modular_stacking_ensemble,
)
from src.models.legacy.nephro_project_plus import (
    run_project_plus_chemberta,
    run_project_plus_gin,
    run_project_plus_gin_hybrid,
)
from src.models.legacy.nephro_sota import run_registered as run_nephro_sota
from src.models.nephro_unsupervised import run_pca_prototype


def _mark_as_prototype(result: ModelResult, *, prototype_name: str, original_name: str) -> ModelResult:
    # -------------------------------------------------------------------------
    # Prototype Relabelling Layer
    # The prototype registry reuses the same evaluation artefacts as the main
    # runners but rewrites their public `model_name` fields. This makes it
    # possible to expose historical baselines alongside the final engines
    # without conflating experimental lineage with the final shortlisted set.
    # -------------------------------------------------------------------------
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for row in result.summary_rows:
        summary_rows.append(
            {
                **row,
                "model_name": prototype_name,
                "notes": combine_notes("prototype_model", f"original_model={original_name}", row.get("notes")),
            }
        )

    for row in result.prediction_rows:
        prediction_rows.append(
            {
                **row,
                "model_name": prototype_name,
            }
        )

    return ModelResult(summary_rows=summary_rows, prediction_rows=prediction_rows)


def run_prototype_nephro_sota(config: dict[str, object]) -> ModelResult:
    # SOTA-inspired CatBoost baseline with descriptor/fingerprint features.
    return _mark_as_prototype(
        run_nephro_sota(config),
        prototype_name="prototype_nephro_sota",
        original_name="nephro_sota",
    )


def run_prototype_nephro_unsupervised_pca(config: dict[str, object]) -> ModelResult:
    # Linear latent baseline retained to contrast with the main autoencoder.
    return _mark_as_prototype(
        run_pca_prototype(config),
        prototype_name="prototype_nephro_unsupervised_pca",
        original_name="nephro_unsupervised::PCA",
    )


def run_prototype_noteworthy_fixed(config: dict[str, object]) -> ModelResult:
    # Archived noteworthy fixed ensemble family.
    return _mark_as_prototype(
        run_noteworthy_fixed(config),
        prototype_name="prototype_noteworthy_fixed",
        original_name="noteworthy_fixed",
    )


def run_prototype_noteworthy_final(config: dict[str, object]) -> ModelResult:
    # Archived noteworthy final family, wrapped for historical traceability.
    return _mark_as_prototype(
        run_noteworthy_final(config),
        prototype_name="prototype_noteworthy_final",
        original_name="noteworthy_final",
    )


def run_prototype_modular_lightgbm(config: dict[str, object]) -> ModelResult:
    # Standalone boosted-tree modular benchmark.
    return _mark_as_prototype(
        run_modular_lightgbm(config),
        prototype_name="prototype_modular_lightgbm",
        original_name="modular_lightgbm",
    )


def run_prototype_modular_histgb(config: dict[str, object]) -> ModelResult:
    # Histogram-based gradient boosting baseline from the modular branch.
    return _mark_as_prototype(
        run_modular_histgb(config),
        prototype_name="prototype_modular_histgb",
        original_name="modular_histgb",
    )


def run_prototype_modular_gin_virtual(config: dict[str, object]) -> ModelResult:
    # Graph neural network prototype retained from the modular search space.
    return _mark_as_prototype(
        run_modular_gin_virtual(config),
        prototype_name="prototype_modular_gin_virtual",
        original_name="modular_gin_virtual",
    )


def run_prototype_modular_node(config: dict[str, object]) -> ModelResult:
    # Neural oblivious decision ensemble prototype.
    return _mark_as_prototype(
        run_modular_node(config),
        prototype_name="prototype_modular_node",
        original_name="modular_node",
    )


def run_prototype_modular_stacking_ensemble(config: dict[str, object]) -> ModelResult:
    # Earlier stacking system predating the final shortlisted ensembles.
    return _mark_as_prototype(
        run_modular_stacking_ensemble(config),
        prototype_name="prototype_modular_stacking_ensemble",
        original_name="modular_stacking_ensemble",
    )


def run_prototype_project_plus_gin(config: dict[str, object]) -> ModelResult:
    # Project-plus graph baseline.
    return _mark_as_prototype(
        run_project_plus_gin(config),
        prototype_name="prototype_project_plus_gin",
        original_name="project_plus_gin",
    )


def run_prototype_project_plus_gin_hybrid(config: dict[str, object]) -> ModelResult:
    # Project-plus hybrid graph/tabular baseline.
    return _mark_as_prototype(
        run_project_plus_gin_hybrid(config),
        prototype_name="prototype_project_plus_gin_hybrid",
        original_name="project_plus_gin_hybrid",
    )


def run_prototype_project_plus_chemberta(config: dict[str, object]) -> ModelResult:
    # Project-plus ChemBERTa prototype from the exploratory lineage.
    return _mark_as_prototype(
        run_project_plus_chemberta(config),
        prototype_name="prototype_project_plus_chemberta",
        original_name="project_plus_chemberta",
    )


PROTOTYPE_MODELS = OrderedDict(
    # Ordered registry so historical baselines appear consistently in reports
    # and optional prototype reruns.
    [
        ("prototype_nephro_sota", run_prototype_nephro_sota),
        ("prototype_nephro_unsupervised_pca", run_prototype_nephro_unsupervised_pca),
        ("prototype_noteworthy_fixed", run_prototype_noteworthy_fixed),
        ("prototype_noteworthy_final", run_prototype_noteworthy_final),
        ("prototype_modular_lightgbm", run_prototype_modular_lightgbm),
        ("prototype_modular_histgb", run_prototype_modular_histgb),
        ("prototype_modular_gin_virtual", run_prototype_modular_gin_virtual),
        ("prototype_modular_node", run_prototype_modular_node),
        ("prototype_modular_stacking_ensemble", run_prototype_modular_stacking_ensemble),
        ("prototype_project_plus_gin", run_prototype_project_plus_gin),
        ("prototype_project_plus_gin_hybrid", run_prototype_project_plus_gin_hybrid),
        ("prototype_project_plus_chemberta", run_prototype_project_plus_chemberta),
    ]
)
