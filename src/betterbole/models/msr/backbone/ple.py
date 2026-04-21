import torch
from torch import nn

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import build_mlp, to_dims
from betterbole.models.utils.container import domain_select


class _PLELayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = num_domains
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.output_dim = expert_dims[-1]

        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                build_mlp(
                    input_dim,
                    expert_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(num_specific_experts)
            ])
            for _ in range(num_domains)
        ])
        self.shared_experts = nn.ModuleList([
            build_mlp(
                input_dim,
                expert_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(num_shared_experts)
        ])
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_specific_experts + num_shared_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_domains)
        ])
        self.shared_gate = nn.Sequential(
            nn.Linear(input_dim, num_domains * num_specific_experts + num_shared_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, task_inputs: list[torch.Tensor], shared_input: torch.Tensor):
        task_outputs = []
        all_specific = []
        shared_outputs = [expert(shared_input) for expert in self.shared_experts]

        for domain_idx in range(self.num_domains):
            task_input = task_inputs[domain_idx]
            specific_outputs = [expert(task_input) for expert in self.specific_experts[domain_idx]]
            all_specific.extend(specific_outputs)
            gate = self.task_gates[domain_idx](task_input).unsqueeze(-1)
            mix_inputs = torch.stack(specific_outputs + shared_outputs, dim=1)
            task_outputs.append(torch.sum(gate * mix_inputs, dim=1))

        shared_gate = self.shared_gate(shared_input).unsqueeze(-1)
        shared_mix = torch.stack(all_specific + shared_outputs, dim=1)
        shared_output = torch.sum(shared_gate * shared_mix, dim=1)
        return task_outputs, shared_output


class PLEBackbone(MSRBackbone):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=expert_dims[-1])
        self.layers = nn.ModuleList()
        layer_input_dim = input_dim
        for _ in range(num_levels):
            layer = _PLELayer(
                input_dim=layer_input_dim,
                num_domains=num_domains,
                expert_dims=expert_dims,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            self.layers.append(layer)
            layer_input_dim = layer.output_dim

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        task_inputs = [x for _ in range(self.num_domains)]
        shared_input = x
        for layer in self.layers:
            task_inputs, shared_input = layer(task_inputs, shared_input)
        stacked = torch.stack(task_inputs, dim=1)
        return domain_select(stacked, domain_ids)


def create_ple(
        input_dim: int = 48,
        num_domains: int = 3,
        batch_size: int = 8,
) -> tuple[PLEBackbone, torch.Tensor]:
    from betterbole.models.utils.tests import dummy_input_multi_domain

    model = PLEBackbone(input_dim=input_dim, num_domains=num_domains)
    x, domain_ids = dummy_input_multi_domain(num_domains=num_domains, batch_size=batch_size, emb_size=input_dim)
    out = model(x, domain_ids)
    assert out.shape == (batch_size, model.output_dim)
    return model, out
