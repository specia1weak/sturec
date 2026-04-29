from __future__ import annotations

from typing import Iterable, List, Union

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.base import BaseModel


def normalize_hidden_dims(values: Iterable[Union[int, str]]) -> List[int]:
    return [int(value) for value in values]


class SimpleTAACModel(BaseModel):
    def __init__(self, manager: SchemaManager, hidden_dims: Iterable[int | str]) -> None:
        super().__init__(manager)
        dims = normalize_hidden_dims(hidden_dims)
        input_dim = self.omni_embedding.whole.embedding_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev_dim, 1)
        self.label_field = manager.label_field

    def _logits(self, interaction) -> torch.Tensor:
        x = self.omni_embedding.whole(interaction)
        x = self.backbone(x)
        return self.head(x).squeeze(-1)

    def predict(self, interaction):
        return torch.sigmoid(self._logits(interaction))

    def calculate_loss(self, interaction):
        labels = interaction[self.label_field].float()
        logits = self._logits(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
