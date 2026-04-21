from typing import Union

import torch
from torch import nn

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import build_mlp, to_dims


class MMoEBackbone(MSRBackbone):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_experts: Union[int, None] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        num_experts = int(num_experts or (num_domains + 1))
        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=expert_dims[-1])
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            build_mlp(
                input_dim,
                expert_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.num_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_domains)
        ])

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_scores = torch.stack([gate(x) for gate in self.gates], dim=1)
        batch_indices = torch.arange(x.size(0), device=x.device)
        gate = gate_scores[batch_indices, domain_ids].unsqueeze(-1)
        return torch.sum(gate * expert_outs, dim=1)


def create_mmoe(
        input_dim: int = 48,
        num_domains: int = 3,
        batch_size: int = 8,
) -> tuple[MMoEBackbone, torch.Tensor]:
    from betterbole.models.utils.tests import dummy_input_multi_domain

    model = MMoEBackbone(input_dim=input_dim, num_domains=num_domains)
    x, domain_ids = dummy_input_multi_domain(num_domains=num_domains, batch_size=batch_size, emb_size=input_dim)
    out = model(x, domain_ids)
    assert out.shape == (batch_size, model.output_dim)
    return model, out
