from typing import Iterable, Type

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.backbone.m3oe import M3oEBackbone, M3oEVersion1Backbone, M3oEVersion2Backbone
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead


class _BaseM3oEModel(MSRModel):
    BACKBONE_CLS: Type[M3oEBackbone] = M3oEBackbone

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            star_dims: Iterable[int] = (512, 256),
            expert_dims: Iterable[int] = (64,),
            num_shared_experts: int = 4,
            factor_mode: str = None,
            shared_gate_detach: bool = None,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        self.backbone = self.BACKBONE_CLS(
            input_dim=self.input_dim,
            num_domains=num_domains,
            star_dims=star_dims,
            expert_dims=expert_dims,
            num_shared_experts=num_shared_experts,
            factor_mode=factor_mode,
            shared_gate_detach=shared_gate_detach,
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


class M3oEModel(_BaseM3oEModel):
    BACKBONE_CLS = M3oEBackbone


class M3oEVersion1Model(_BaseM3oEModel):
    BACKBONE_CLS = M3oEVersion1Backbone


class M3oEVersion2Model(_BaseM3oEModel):
    BACKBONE_CLS = M3oEVersion2Backbone
