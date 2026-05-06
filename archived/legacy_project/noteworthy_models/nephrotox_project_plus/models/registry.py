from .chemberta import ChemBERTaClassifier
from .gin import GINClassifier
from .gin_hybrid import GINHybridClassifier

MODEL_REGISTRY = {
    "gin": {"class": GINClassifier, "family": "graph"},
    "gin_hybrid": {"class": GINHybridClassifier, "family": "graph"},
    "chemberta": {"class": ChemBERTaClassifier, "family": "text"},
}



def get_model_class(name: str):
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]["class"]



def get_model_family(name: str) -> str:
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]["family"]
