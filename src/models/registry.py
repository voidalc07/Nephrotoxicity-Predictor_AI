from __future__ import annotations

from collections import OrderedDict
from typing import Callable

from src.models.common import ModelResult
from src.models.nephro_chemberta import run_registered as run_nephro_chemberta
from src.models.nephro_full_coverage import (
    run_full_coverage_descriptor_fp_meta_v4,
    run_full_coverage_improvements,
)
from src.models.nephro_tanimoto_gpc import run_registered as run_nephro_tanimoto_gpc
from src.models.nephro_unsupervised import run_registered as run_nephro_unsupervised
from src.models.prototypes import PROTOTYPE_MODELS

ModelRunner = Callable[[dict[str, object]], ModelResult]

# -------------------------------------------------------------------------
# Registered Model Families
# The registry is the experiment manifest for the portable project. It maps
# stable model identifiers to executable runners, preserving the five main
# dashboard engines while also exposing prototype and legacy branches for
# comparative methodology work.
# -------------------------------------------------------------------------
MAIN_MODELS: "OrderedDict[str, ModelRunner]" = OrderedDict(
    [
        ("nephro_chemberta", run_nephro_chemberta),
        ("nephro_unsupervised", run_nephro_unsupervised),
        ("full_coverage_improvements", run_full_coverage_improvements),
        ("full_coverage_descriptor_fp_meta_v4", run_full_coverage_descriptor_fp_meta_v4),
        ("modular_tanimoto_gpc", run_nephro_tanimoto_gpc),
    ]
)

# -------------------------------------------------------------------------
# Combined Registry
# ALL_MODELS extends the final engine set with prototype families so the
# command-line pipeline can reproduce exploratory comparisons without mixing
# those provisional results into the default production benchmark.
# -------------------------------------------------------------------------
ALL_MODELS: "OrderedDict[str, ModelRunner]" = OrderedDict()
ALL_MODELS.update(MAIN_MODELS)
ALL_MODELS.update(PROTOTYPE_MODELS)

MODELS = MAIN_MODELS
