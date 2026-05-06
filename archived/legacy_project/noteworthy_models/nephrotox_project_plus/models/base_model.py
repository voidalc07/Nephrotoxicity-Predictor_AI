from abc import ABC, abstractmethod
import torch.nn as nn


class BaseGraphModel(nn.Module, ABC):
    is_graph_model = True

    @abstractmethod
    def forward(self, x, edge_index, batch, descriptors=None):
        raise NotImplementedError


class BaseTextModel(nn.Module, ABC):
    is_graph_model = False

    @abstractmethod
    def forward(self, input_ids, attention_mask):
        raise NotImplementedError
