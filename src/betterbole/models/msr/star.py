from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.common import default_dims, to_dims
from betterbole.models.utils.container import domain_select


class StarExpert(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        if len(dims) == 1 and isinstance(dims[0], Iterable):
            dims = tuple(dims[0])
        self.dims = tuple(int(dim) for dim in dims)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:]):
            self.weights.append(nn.Parameter(torch.randn(in_dim, out_dim) / (in_dim ** 0.5)))
            self.biases.append(nn.Parameter(torch.zeros(out_dim)))

    def merge_with(self, other: "StarExpert"):
        merged_weights = [w1 * w2 for w1, w2 in zip(self.weights, other.weights)]
        merged_biases = [b1 + b2 for b1, b2 in zip(self.biases, other.biases)]
        return merged_weights, merged_biases

    @staticmethod
    def forward_with_params(weights, biases, x, activation: str = "relu"):
        activation_fn = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
        }.get(activation.lower(), F.relu)
        out = x
        for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
            out = out @ weight + bias
            if layer_idx != len(weights) - 1:
                out = activation_fn(out)
        return out


class STARBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            activation: str = "relu",
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, default_dims(input_dim))
        linear_stream = (input_dim, *expert_dims)
        self.output_dim = int(expert_dims[-1])
        self.activation = activation
        self.shared_expert = StarExpert(*linear_stream)
        self.domain_experts = nn.ModuleList([
            StarExpert(*linear_stream) for _ in range(num_domains)
        ])
        self._reset_star_parameters()

    def _reset_star_parameters(self):
        for weight in self.shared_expert.weights:
            nn.init.xavier_uniform_(weight)
        for bias in self.shared_expert.biases:
            nn.init.zeros_(bias)

        for expert in self.domain_experts:
            for weight in expert.weights:
                nn.init.normal_(weight, mean=1.0, std=0.02)
            for bias in expert.biases:
                nn.init.zeros_(bias)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        merged = [self.shared_expert.merge_with(domain_expert) for domain_expert in self.domain_experts]
        outputs = [
            StarExpert.forward_with_params(weights, biases, x, activation=self.activation)
            for weights, biases in merged
        ]
        return domain_select(torch.stack(outputs, dim=1), domain_ids)


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
