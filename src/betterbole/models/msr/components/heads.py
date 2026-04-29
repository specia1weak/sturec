from typing import Iterable

import torch
from torch import nn

from betterbole.models.msr.backbone.common import to_dims
from betterbole.models.utils.container import MultiScenarioContainer
from betterbole.models.utils.general import MLP


class DomainTowerHead(nn.Module):
    def __init__(
            self,
            num_domains: int,
            input_dim: int,
            hidden_dims: Iterable[int] = None,
            dropout_rate: float = 0.2,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        default_hidden = (max(1, input_dim // 2),)
        tower_hidden_dims = to_dims(hidden_dims, default_hidden)
        self.head = MultiScenarioContainer(
            num_domains,
            lambda: MLP(
                input_dim,
                *tower_hidden_dims,
                1,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(x, domain_ids).squeeze(-1)
