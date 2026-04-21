from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import to_dims
from betterbole.models.utils.container import domain_select
from betterbole.models.utils.tests import dummy_input_multi_domain


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


class STARBackbone(MSRBackbone):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            activation: str = "relu",
    ):
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        linear_stream = (input_dim, *expert_dims)
        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=expert_dims[-1])
        self.activation = activation
        self.shared_expert = StarExpert(*linear_stream)
        self.domain_experts = nn.ModuleList([
            StarExpert(*linear_stream) for _ in range(num_domains)
        ])

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        merged = [self.shared_expert.merge_with(domain_expert) for domain_expert in self.domain_experts]
        outputs = [
            StarExpert.forward_with_params(weights, biases, x, activation=self.activation)
            for weights, biases in merged
        ]
        return domain_select(torch.stack(outputs, dim=1), domain_ids)


def create_star(
        input_dim: int = 48,
        num_domains: int = 3,
        batch_size: int = 8,
) -> tuple[STARBackbone, torch.Tensor]:
    model = STARBackbone(input_dim=input_dim, num_domains=num_domains)
    x, domain_ids = dummy_input_multi_domain(num_domains=num_domains, batch_size=batch_size, emb_size=input_dim)
    out = model(x, domain_ids)
    assert out.shape == (batch_size, model.output_dim)
    print(
        "[create_star]",
        {
            "x_shape": tuple(x.shape),
            "domain_ids_shape": tuple(domain_ids.shape),
            "output_shape": tuple(out.shape),
            "output_dim": model.output_dim,
        },
    )
    return model, out


if __name__ == "__main__":
    create_star()
