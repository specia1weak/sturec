from __future__ import annotations

import torch
from torch import nn

from betterbole.models.msr.pepnet.blocks import PPBlock
from betterbole.models.utils.common import build_mlp, to_dims
from betterbole.models.utils.container import domain_select


class PPNetBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims=None,
            scenario_dim=None,
            dropout_rate: float = 0.0,
    ):
        hidden_dims = to_dims(hidden_dims, (input_dim, input_dim))
        super().__init__()
        self.output_dim = int(hidden_dims[-1])
        self.num_domains = int(num_domains)
        self.scenario_dim = int(scenario_dim or max(1, input_dim // 3))
        self.agnostic_dim = input_dim - self.scenario_dim
        if self.agnostic_dim <= 0:
            raise ValueError("scenario_dim must be smaller than input_dim")
        self.domain_blocks = nn.ModuleList([
            PPBlock(
                agnostic_dim=self.agnostic_dim,
                gate_input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_domains)
        ])

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        agnostic_x = x[:, self.scenario_dim:]
        gate_input = torch.cat([x[:, :self.scenario_dim], agnostic_x.detach()], dim=-1)
        outputs = torch.stack([block(agnostic_x, gate_input) for block in self.domain_blocks], dim=1)
        return domain_select(outputs, domain_ids.long())


class EPNetBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims=None,
            scenario_dim=None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        hidden_dims = to_dims(hidden_dims, (input_dim, input_dim))
        super().__init__()
        self.output_dim = int(hidden_dims[-1])
        self.num_domains = int(num_domains)
        self.scenario_dim = int(scenario_dim or max(1, input_dim // 3))
        self.agnostic_dim = input_dim - self.scenario_dim
        if self.agnostic_dim <= 0:
            raise ValueError("scenario_dim must be smaller than input_dim")
        self.gate = nn.Sequential(
            nn.Linear(input_dim, self.agnostic_dim),
            nn.Sigmoid(),
        )
        self.agnostic_project = build_mlp(
            self.agnostic_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        del domain_ids
        scenario_x = x[:, :self.scenario_dim]
        agnostic_x = x[:, self.scenario_dim:]
        gate_input = torch.cat([scenario_x, agnostic_x.detach()], dim=-1)
        gated = agnostic_x * self.gate(gate_input)
        return self.agnostic_project(gated)
