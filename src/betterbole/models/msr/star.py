from typing import Iterable

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.backbone.star import STARBackbone
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead


class STARModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (256, 128),
            activation: str = "relu",
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        self.backbone = STARBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            activation=activation,
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
