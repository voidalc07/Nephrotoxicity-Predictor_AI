from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PROCESSED_DATA_DIR = PROJECT_ROOT / "datasets" / "processed"


@dataclass
class Config:
    model_name: str = "gin"
    train_csv: str = str(PROCESSED_DATA_DIR / "model_construction_dataset.csv")
    test_csv: str = str(PROCESSED_DATA_DIR / "external_test_dataset.csv")
    label_col: str = "label"
    scaffold_type: str = "murcko"  # murcko | carbon
    val_fraction: float = 0.20
    batch_size: int = 64
    hidden_dim: int = 96
    num_layers: int = 4
    dropout: float = 0.35
    lr: float = 8e-4
    weight_decay: float = 2e-4
    epochs: int = 60
    patience: int = 10
    seed: int = 42
    prefer_mps: bool = True
    output_dir: str = "outputs"

    # Hybrid GIN
    descriptor_dim: int = 12

    # ChemBERTa
    hf_model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    max_length: int = 128
    freeze_backbone: bool = True
    unfreeze_last_n_layers: int = 2
    grad_clip: float = 1.0

    def to_dict(self):
        return asdict(self)
