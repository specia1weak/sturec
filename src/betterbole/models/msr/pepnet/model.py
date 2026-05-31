from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.msr.pepnet.backbones import EPNetBackbone, PPNetBackbone


class _PEPNetBaseModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (128, 64),
            scenario_dim: int = None,
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

        self.backbone = self.build_backbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            scenario_dim=scenario_dim,
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
    def build_backbone(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Iterable[int],
            scenario_dim: Optional[int],
            dropout_rate: float,
            activation: str,
            batch_norm: bool,
    ) -> nn.Module:
        raise NotImplementedError

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
        return F.binary_cross_entropy_with_logits(logits, labels)


class PPNetModel(_PEPNetBaseModel):
    def build_backbone(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Iterable[int],
            scenario_dim: Optional[int],
            dropout_rate: float,
            activation: str,
            batch_norm: bool,
    ) -> nn.Module:
        del activation, batch_norm
        return PPNetBackbone(
            input_dim=input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            scenario_dim=scenario_dim,
            dropout_rate=dropout_rate,
        )


class EPNetModel(_PEPNetBaseModel):
    def build_backbone(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Iterable[int],
            scenario_dim: Optional[int],
            dropout_rate: float,
            activation: str,
            batch_norm: bool,
    ) -> nn.Module:
        return EPNetBackbone(
            input_dim=input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            scenario_dim=scenario_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )


class PEPNetModel(EPNetModel):
    pass
