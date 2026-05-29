from typing import Iterable, Optional

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.common import build_mlp, to_dims


class MMoEBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_experts: Optional[int] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        num_experts = int(num_experts or (num_domains + 1))
        self.output_dim = int(expert_dims[-1])
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


class MMoEModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_experts: int = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        self.backbone = MMoEBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_experts=num_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x, domain_ids), domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
