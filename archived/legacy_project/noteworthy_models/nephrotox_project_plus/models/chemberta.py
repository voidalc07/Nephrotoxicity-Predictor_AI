from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .base_model import BaseTextModel


class ChemBERTaClassifier(BaseTextModel):
    def __init__(
        self,
        hf_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        dropout: float = 0.25,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hf_model_name = hf_model_name
        self.config = AutoConfig.from_pretrained(hf_model_name)
        self.backbone = AutoModel.from_pretrained(hf_model_name, add_pooling_layer=False)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            encoder = getattr(self.backbone, "encoder", None)
            if encoder is not None and hasattr(encoder, "layer") and unfreeze_last_n_layers > 0:
                for layer in encoder.layer[-unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            embeddings = getattr(self.backbone, "embeddings", None)
            if embeddings is not None:
                for param in embeddings.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_token)).view(-1)
        return logits
