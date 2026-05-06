import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm
from .base_model import BaseGraphModel


class GINClassifier(BaseGraphModel):
    def __init__(self, input_dim: int, hidden_dim: int = 96, num_layers: int = 4, dropout: float = 0.35, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(BatchNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_graph(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_add_pool(x, batch)

    def forward(self, x, edge_index, batch, descriptors=None):
        graph_emb = self.encode_graph(x, edge_index, batch)
        logits = self.head(graph_emb).view(-1)
        return logits
