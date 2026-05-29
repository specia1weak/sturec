from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import EmbView, OmniEmbLayer
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.common import build_mlp, to_dims
from betterbole.models.utils.container import domain_select


class CrocodileMMoELayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            prior_input_dim: int,
            num_experts: int,
            num_domains: int,
            expert_dims,
            gate_hidden_dims,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (256, 128, 64))
        gate_hidden_dims = to_dims(gate_hidden_dims, (128, 64))
        self.num_experts = int(num_experts)
        self.num_domains = int(num_domains)
        self.output_dim = int(expert_dims[-1])

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
            build_mlp(
                prior_input_dim,
                (*gate_hidden_dims, self.num_experts * self.output_dim),
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_domains)
        ])

    def forward(self, expert_inputs: list[torch.Tensor], prior_x: torch.Tensor):
        expert_outputs = torch.stack(
            [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)],
            dim=1,
        )

        mixed_outputs = []
        gate_outputs = []
        flat_prior = prior_x.flatten(start_dim=1)
        for gate in self.gates:
            gate_output = gate(flat_prior).reshape(-1, self.num_experts, self.output_dim)
            gate_output = torch.softmax(gate_output, dim=1)
            gate_outputs.append(gate_output)
            mixed_outputs.append(torch.sum(gate_output * expert_outputs, dim=1))
        gate_outputs = torch.stack(gate_outputs, dim=-1)
        return mixed_outputs, gate_outputs, expert_outputs


class CrocodileModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            num_experts: int = None,
            expert_dims: Iterable[int] = (256, 128, 64),
            gate_hidden_dims: Iterable[int] = (128, 64),
            tower_hidden_dims: Iterable[int] = (128, 64),
            prior_fields: Iterable[str] = None,
            disentangled_weight: float = 1e-3,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.num_experts = int(num_experts or num_domains)
        self.disentangled_weight = float(disentangled_weight)

        resolved_prior_fields = self._resolve_prior_fields(prior_fields)
        self.prior_view = EmbView(self.omni_embedding, include_fields=resolved_prior_fields)
        self.prior_dim = self.prior_view.embedding_dim
        if self.prior_dim <= 0:
            raise ValueError("CrocodileModel requires at least one prior field with embedding dimension > 0")

        self.expert_omni_layers = nn.ModuleList(
            [OmniEmbLayer(manager=manager) for _ in range(max(0, self.num_experts - 1))]
        )
        self.expert_views = [self.omni_embedding.whole_without_domain] + [
            layer.whole_without_domain for layer in self.expert_omni_layers
        ]
        self.input_dim = self.expert_views[0].embedding_dim
        if self.input_dim <= 0:
            raise ValueError("CrocodileModel requires non-empty expert input embeddings")

        self.crocodile = CrocodileMMoELayer(
            input_dim=self.input_dim,
            prior_input_dim=self.prior_dim,
            num_experts=self.num_experts,
            num_domains=num_domains,
            expert_dims=expert_dims,
            gate_hidden_dims=gate_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.crocodile.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )
        self._latest_gate_outputs = None
        self._latest_expert_outputs = None

    def _resolve_prior_fields(self, prior_fields) -> tuple[str, ...]:
        if prior_fields is not None:
            if isinstance(prior_fields, (str, bytes)):
                prior_fields = (prior_fields,)
            return tuple(prior_fields)

        defaults = []
        for field_name in (self.manager.uid_field, self.manager.iid_field, self.manager.domain_field):
            if field_name is not None:
                defaults.append(field_name)
        if not defaults:
            raise ValueError("Could not infer default prior_fields for CrocodileModel")
        return tuple(defaults)

    def encode_features(self, interaction):
        expert_inputs = [
            torch.flatten(view(interaction), start_dim=1)
            for view in self.expert_views
        ]
        prior_x = self.prior_view(interaction)
        domain_ids = interaction[self.DOMAIN].long()
        return expert_inputs, prior_x, domain_ids

    def forward(self, expert_inputs: list[torch.Tensor], prior_x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        tower_inputs, gate_outputs, expert_outputs = self.crocodile(expert_inputs, prior_x)
        self._latest_gate_outputs = gate_outputs
        self._latest_expert_outputs = expert_outputs
        selected_input = domain_select(torch.stack(tower_inputs, dim=1), domain_ids)
        return self.head(selected_input, domain_ids)

    def predict(self, interaction):
        expert_inputs, prior_x, domain_ids = self.encode_features(interaction)
        return self.forward(expert_inputs, prior_x, domain_ids)

    def _covariance_regularization(self) -> torch.Tensor:
        if self._latest_expert_outputs is None:
            return torch.tensor(0.0, device=self.head.head.domain_networks[0].net[0].weight.device)

        expert_outputs = self._latest_expert_outputs
        disentangled_loss = expert_outputs.new_tensor(0.0)
        for i in range(self.num_experts - 1):
            centered_i = expert_outputs[:, i, :] - expert_outputs[:, i, :].mean(dim=0, keepdim=True)
            for j in range(i, self.num_experts):
                centered_j = expert_outputs[:, j, :] - expert_outputs[:, j, :].mean(dim=0, keepdim=True)
                disentangled_loss = disentangled_loss + torch.mean(
                    torch.abs(torch.matmul(centered_i.transpose(0, 1), centered_j))
                )
        return disentangled_loss

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return bce_loss + self.disentangled_weight * self._covariance_regularization()
