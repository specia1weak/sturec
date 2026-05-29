from typing import Iterable

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.general import MLP


class SharedFeatureGate(nn.Module):
    def __init__(
            self,
            input_dim: int,
            gate_hidden_dims=None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            residual_scale: float = 1.0,
    ):
        super().__init__()
        gate_hidden_dims = to_dims(gate_hidden_dims, (max(1, input_dim // 2),))
        self.residual_scale = float(residual_scale)
        self.gate = MLP(
            input_dim,
            *gate_hidden_dims,
            input_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)
        gate = torch.sigmoid(gate_logits)
        scaled = x * (1.0 + self.residual_scale * gate)
        return scaled, gate


class DomainFeatureGate(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128),
            gate_hidden_dims=None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            gate_dropout_rate: float = 0.0,
            gate_activation: str = "relu",
            gate_batch_norm: bool = False,
            gate_residual_scale: float = 1.0,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__()
        hidden_dims = to_dims(hidden_dims, (256, 128))
        self.gate = SharedFeatureGate(
            input_dim=input_dim,
            gate_hidden_dims=gate_hidden_dims,
            dropout_rate=gate_dropout_rate,
            activation=gate_activation,
            batch_norm=gate_batch_norm,
            residual_scale=gate_residual_scale,
        )
        self.bottom = MLP(
            input_dim,
            *hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.output_dim = int(hidden_dims[-1])
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated_x, gate = self.gate(x)
        features = self.bottom(gated_x)
        logits = self.head(features, domain_ids)
        return logits, gate


class FeatureGateModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            gate_hidden_dims=None,
            gate_dropout_rate: float = 0.0,
            gate_activation: str = "relu",
            gate_batch_norm: bool = False,
            gate_residual_scale: float = 1.0,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim
        self.encoder = DomainFeatureGate(
            input_dim=self.input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            gate_hidden_dims=gate_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            gate_dropout_rate=gate_dropout_rate,
            gate_activation=gate_activation,
            gate_batch_norm=gate_batch_norm,
            gate_residual_scale=gate_residual_scale,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
        )
        self._latest_gate: torch.Tensor | None = None

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        logits, gate = self.encoder(x, domain_ids)
        self._latest_gate = gate.detach()
        return logits

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
