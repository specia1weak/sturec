from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import EmbView, OmniEmbLayer, RecEmbedding
from betterbole.models.msr.base import MSRModel
from betterbole.models.utils.activation import activation_layer
from betterbole.models.utils.common import to_dims


class ReferenceMLPBlock(nn.Module):
    """
    Match the reference MLP_Block semantics:
    - hidden_units define hidden layers
    - optional explicit output_dim appends one last Linear without activation
    - activation/dropout/batch_norm apply only to hidden layers
    """

    def __init__(
            self,
            input_dim: int,
            hidden_units,
            output_dim: int = None,
            hidden_activations: str = "relu",
            dropout_rates: float = 0.0,
            batch_norm: bool = False,
    ):
        super().__init__()
        hidden_units = to_dims(hidden_units, ())
        dims = [int(input_dim), *[int(v) for v in hidden_units]]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation_layer(hidden_activations))
            if float(dropout_rates) > 0.0:
                layers.append(nn.Dropout(float(dropout_rates)))

        if output_dim is not None:
            last_in_dim = dims[-1]
            layers.append(nn.Linear(last_in_dim, int(output_dim)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrocodileV1ExpertEmbedding(nn.Module):
    """
    Build the reference expert embedding path:
    every non-domain non-sequence field is embedded to base_dim * num_experts,
    then the final tensor is split by expert on the last dimension.
    """

    def __init__(self, manager: SchemaManager, num_experts: int):
        super().__init__()
        self.manager = manager
        self.num_experts = int(num_experts)
        self.domain_fields = set(manager.domain_fields)

        self.feature_settings = []
        self.emb_modules = nn.ModuleDict()
        self.base_embedding_dim = None

        for setting in manager.settings:
            if setting.is_sequence_setting:
                continue
            if setting.field_name in self.domain_fields:
                continue
            self.feature_settings.append(setting)

            base_dim = int(setting.embedding_dim)
            if self.base_embedding_dim is None:
                self.base_embedding_dim = base_dim
            elif self.base_embedding_dim != base_dim:
                raise ValueError(
                    "CrocodileV1 requires uniform feature embedding_dim across expert-input fields "
                    "to match the reference split-by-expert implementation."
                )

            if setting.requires_embedding_module:
                self.emb_modules[setting.field_name] = RecEmbedding(
                    num_embeddings=setting.num_embeddings,
                    embedding_dim=base_dim * self.num_experts,
                    padding_idx={True: 0, False: None}[setting.padding_zero],
                )
            else:
                self.emb_modules[setting.field_name] = nn.Linear(
                    base_dim,
                    base_dim * self.num_experts,
                    bias=False,
                )

        if not self.feature_settings:
            raise ValueError("CrocodileV1 requires at least one expert-input feature")
        if self.base_embedding_dim is None:
            raise ValueError("CrocodileV1 could not infer base embedding_dim")

        self.num_fields = len(self.feature_settings)
        self.output_dim = self.base_embedding_dim * self.num_fields

    def forward(self, interaction) -> torch.Tensor:
        feature_embs = []
        for setting in self.feature_settings:
            emb = setting.compute_tensor(interaction, self.emb_modules)
            feature_embs.append(emb)
        return torch.stack(feature_embs, dim=1)


class CrocodileV1MMoELayer(nn.Module):
    def __init__(
            self,
            num_experts: int,
            num_domains: int,
            embedding_dim: int,
            input_dim: int,
            prior_input_dim: int,
            expert_hidden_units,
            gate_hidden_units,
            hidden_activations: str,
            net_dropout: float,
            batch_norm: bool,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.num_domains = int(num_domains)
        self.embedding_dim = int(embedding_dim)
        self.expert_hidden_units = to_dims(expert_hidden_units, (512, 256, 128))
        gate_hidden_units = to_dims(gate_hidden_units, (128, 64))

        self.experts = nn.ModuleList([
            ReferenceMLPBlock(
                input_dim=input_dim,
                hidden_units=self.expert_hidden_units,
                hidden_activations=hidden_activations,
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_experts)
        ])
        self.gate = nn.ModuleList([
            ReferenceMLPBlock(
                input_dim=prior_input_dim,
                hidden_units=gate_hidden_units,
                output_dim=self.num_experts * self.expert_hidden_units[-1],
                hidden_activations=hidden_activations,
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_domains)
        ])

    def forward(self, x: torch.Tensor, prior_emb: torch.Tensor):
        split_x = x.split(self.embedding_dim, dim=2)
        expert_inputs = [emb.flatten(start_dim=1) for emb in split_x]
        experts_output = torch.stack(
            [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)],
            dim=1,
        )

        mmoe_output = []
        gate_output_list = []
        flat_prior = prior_emb.flatten(start_dim=1)
        for i in range(self.num_domains):
            gate_output = self.gate[i](flat_prior)
            gate_output = gate_output.reshape(-1, self.num_experts, self.expert_hidden_units[-1])
            gate_output = torch.softmax(gate_output, dim=1)
            gate_output_list.append(gate_output)
            mmoe_output.append(torch.sum(torch.mul(gate_output, experts_output), dim=1))
        gate_output_list = torch.stack(gate_output_list, dim=-1)
        return mmoe_output, gate_output_list, experts_output


class CrocodileV1Model(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            num_experts: int = None,
            expert_hidden_units: Iterable[int] = (256, 128, 64),
            gate_hidden_units: Iterable[int] = (128, 64),
            tower_hidden_units: Iterable[int] = (128, 64),
            hidden_activations: str = "relu",
            net_dropout: float = 0.0,
            batch_norm: bool = False,
            embedding_regularizer: float = 0.0,
            net_regularizer: float = 1e-6,
            disentangled_weight: float = 1e-3,
            prior_idx: Iterable[str] = None,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.num_experts = int(num_experts or num_domains)
        self.num_domains = int(num_domains)
        self.disentangled_weight = float(disentangled_weight)
        self.embedding_regularizer = float(embedding_regularizer)
        self.net_regularizer = float(net_regularizer)

        self.embedding_layer = CrocodileV1ExpertEmbedding(manager, num_experts=self.num_experts)
        self.embedding_dim = int(self.embedding_layer.base_embedding_dim)

        resolved_prior_idx = self._resolve_prior_idx(prior_idx)
        self.prior_embedding_layer = OmniEmbLayer(manager=manager)
        self.prior_view = EmbView(self.prior_embedding_layer, include_fields=resolved_prior_idx)
        self.prior_input_dim = self.prior_view.embedding_dim
        if self.prior_input_dim <= 0:
            raise ValueError("CrocodileV1 requires at least one prior field")

        self.mmoe_layer = CrocodileV1MMoELayer(
            num_experts=self.num_experts,
            num_domains=self.num_domains,
            embedding_dim=self.embedding_dim,
            input_dim=self.embedding_dim * self.embedding_layer.num_fields,
            prior_input_dim=self.prior_input_dim,
            expert_hidden_units=expert_hidden_units,
            gate_hidden_units=gate_hidden_units,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm,
        )
        tower_hidden_units = to_dims(tower_hidden_units, (128, 64))
        self.tower = nn.ModuleList([
            ReferenceMLPBlock(
                input_dim=self.mmoe_layer.expert_hidden_units[-1],
                hidden_units=tower_hidden_units,
                output_dim=1,
                hidden_activations=hidden_activations,
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_domains)
        ])

    def _resolve_prior_idx(self, prior_idx) -> tuple[str, ...]:
        if prior_idx is not None:
            if isinstance(prior_idx, (str, bytes)):
                prior_idx = (prior_idx,)
            return tuple(prior_idx)

        defaults = []
        for field_name in (self.manager.uid_field, self.manager.iid_field, self.manager.domain_field):
            if field_name is not None:
                defaults.append(field_name)
        if not defaults:
            raise ValueError("Could not infer default prior_idx for CrocodileV1")
        return tuple(defaults)

    def encode_features(self, interaction):
        x = self.embedding_layer(interaction)
        prior_emb = self.prior_view(interaction)
        domain_ids = interaction[self.DOMAIN].long()
        return x, prior_emb, domain_ids

    def _forward_dict(self, x: torch.Tensor, prior_emb: torch.Tensor, domain_ids: torch.Tensor):
        tower_inputs, gate_outputs, expert_outputs = self.mmoe_layer(x, prior_emb)
        tower_output = [self.tower[i](tower_inputs[i]) for i in range(self.num_domains)]
        indices = (torch.arange(0, len(domain_ids), device=domain_ids.device) * self.num_domains + domain_ids).long()
        logits = torch.stack(tower_output, dim=1)
        logits_flattened = torch.flatten(logits, start_dim=0, end_dim=1)
        selected_logits = torch.index_select(logits_flattened, 0, indices).squeeze(-1)
        return {
            "gate_outputs": gate_outputs,
            "expert_output": expert_outputs,
            "logits": selected_logits,
        }

    def forward(self, x: torch.Tensor, prior_emb: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, prior_emb, domain_ids)["logits"]

    def predict(self, interaction):
        x, prior_emb, domain_ids = self.encode_features(interaction)
        return torch.sigmoid(self.forward(x, prior_emb, domain_ids))

    def _covariance_regularization(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        disentangled_loss = expert_outputs.new_tensor(0.0)
        for i in range(self.num_experts - 1):
            for j in range(i, self.num_experts):
                disentangled_loss += torch.mean(
                    torch.abs(torch.matmul(
                        (expert_outputs[:, i, :] - torch.mean(expert_outputs[:, i, :], dim=0, keepdim=True)).T,
                        (expert_outputs[:, j, :] - torch.mean(expert_outputs[:, j, :], dim=0, keepdim=True))
                    ))
                )
        return disentangled_loss

    def regularization_loss(self) -> torch.Tensor:
        reg_term = torch.zeros((), device=next(self.parameters()).device)
        if self.embedding_regularizer <= 0.0 and self.net_regularizer <= 0.0:
            return reg_term

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "embedding_layer" in name:
                if self.embedding_regularizer > 0.0:
                    reg_term = reg_term + self.embedding_regularizer * torch.sum(param * param)
            else:
                if self.net_regularizer > 0.0:
                    reg_term = reg_term + self.net_regularizer * torch.sum(param * param)
        return reg_term

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        x, prior_emb, domain_ids = self.encode_features(interaction)
        return_dict = self._forward_dict(x, prior_emb, domain_ids)
        probs = torch.sigmoid(return_dict["logits"])
        bce_loss = F.binary_cross_entropy(probs, labels)
        return (
            bce_loss
            + self.disentangled_weight * self._covariance_regularization(return_dict["expert_output"])
            + self.regularization_loss()
        )
