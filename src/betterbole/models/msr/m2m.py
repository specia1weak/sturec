from typing import Iterable

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.common import build_mlp, to_dims


class M2MBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_experts: int = 4,
            domain_emb_dim=None,
            nhead: int = 4,
            num_encoder_layers: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        domain_emb_dim = int(domain_emb_dim or expert_dims[-1])
        self.output_dim = int(expert_dims[-1])
        self.num_experts = int(num_experts)
        self.domain_embedding = nn.Embedding(num_domains, domain_emb_dim)
        self.input_project = nn.Linear(input_dim, input_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=max(input_dim * 2, expert_dims[-1]),
                dropout=dropout_rate,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
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
        self.domain_project = build_mlp(
            domain_emb_dim,
            (expert_dims[-1],),
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=False,
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(expert_dims[-1] * 2, self.num_experts),
            nn.Softmax(dim=-1),
        )
        self.meta_tower = nn.Sequential(
            nn.Linear(expert_dims[-1] * 2, expert_dims[-1]),
            nn.ReLU(),
            nn.Linear(expert_dims[-1], expert_dims[-1]),
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        transformed = self.input_project(x).unsqueeze(1)
        transformed = self.transformer(transformed).squeeze(1)
        expert_outs = torch.stack([expert(transformed) for expert in self.experts], dim=1)
        domain_state = self.domain_project(self.domain_embedding(domain_ids.long()))
        pooled_hint = expert_outs.mean(dim=1)
        gate = self.meta_gate(torch.cat([pooled_hint, domain_state], dim=-1)).unsqueeze(-1)
        mixed = torch.sum(gate * expert_outs, dim=1)
        return self.meta_tower(torch.cat([mixed, domain_state], dim=-1)) + mixed


class M2MModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_experts: int = 4,
            domain_emb_dim: int = None,
            nhead: int = 4,
            num_encoder_layers: int = 1,
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

        self.backbone = M2MBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_experts=num_experts,
            domain_emb_dim=domain_emb_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
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
