import torch
import torch.nn as nn
from .gin import GINClassifier


class GINHybridClassifier(GINClassifier):
    def __init__(self, input_dim: int, hidden_dim: int = 96, num_layers: int = 4, dropout: float = 0.35, descriptor_dim: int = 12, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.descriptor_tower = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch, descriptors=None):
        if descriptors is None:
            raise ValueError("GINHybridClassifier requires descriptors in forward().")
        graph_emb = self.encode_graph(x, edge_index, batch)
        desc_emb = self.descriptor_tower(descriptors)
        fused = torch.cat([graph_emb, desc_emb], dim=1)
        logits = self.fusion_head(fused).view(-1)
        return logits
