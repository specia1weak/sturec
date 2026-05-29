import torch
from torch import nn

from betterbole.models.utils.common import build_mlp, to_dims
from betterbole.models.utils.container import domain_select


class PLELayer(nn.Module):
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


class PLEFeatureGateLayer(nn.Module):
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
            detach_gate_input: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = num_domains
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.output_dim = expert_dims[-1]
        self.detach_gate_input = bool(detach_gate_input)

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
        self.specific_gates = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    build_mlp(
                        input_dim,
                        (*expert_dims[:-1], num_shared_experts * self.output_dim),
                        dropout_rate=dropout_rate,
                        activation=activation,
                        batch_norm=batch_norm,
                    ),
                    nn.Sigmoid(),
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

    def forward(self, task_inputs: list[torch.Tensor], shared_input: torch.Tensor):
        shared_outputs = torch.stack([expert(shared_input) for expert in self.shared_experts], dim=1)
        shared_output = shared_outputs.sum(dim=1)

        task_outputs = []
        for domain_idx in range(self.num_domains):
            task_input = task_inputs[domain_idx]
            gate_input = task_input.detach() if self.detach_gate_input else task_input
            mixed_specific_outputs = []
            for expert, gate in zip(self.specific_experts[domain_idx], self.specific_gates[domain_idx]):
                specific_output = expert(task_input)
                gate_output = gate(gate_input).reshape(-1, self.num_shared_experts, self.output_dim)
                gated_shared_output = (gate_output * shared_outputs).sum(dim=1)
                mixed_specific_outputs.append(specific_output + gated_shared_output)
            task_outputs.append(torch.stack(mixed_specific_outputs, dim=0).sum(dim=0))
        return task_outputs, shared_output


def select_domain_output(task_outputs: list[torch.Tensor], domain_ids: torch.Tensor) -> torch.Tensor:
    return domain_select(torch.stack(task_outputs, dim=1), domain_ids)
