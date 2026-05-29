from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.msr.ple.layers import select_domain_output
from betterbole.models.utils.common import build_mlp, to_dims
from betterbole.models.utils.container import domain_select


class RIPLESharedExpert(nn.Module):
    """
    原版 RIPLE 逻辑：共享主干 + domain-specific residual expert + 解耦约束。
    这里把它包装成 PLE 的 shared expert。
    """

    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            margin: float = 0.1,
            aux_loss_weight: float = 0.1,
    ):
        super().__init__()
        hidden_dims = to_dims(hidden_dims, (input_dim, input_dim))
        self.backbone = build_mlp(
            input_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.output_dim = int(hidden_dims[-1])
        self.num_domains = int(num_domains)
        self.margin = float(margin)
        self.aux_loss_weight = float(aux_loss_weight)
        self._cos_sim: torch.Tensor | None = None

        self.residual_experts = nn.ModuleList([
            build_mlp(
                input_dim + self.output_dim,
                (input_dim, self.output_dim),
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_domains)
        ])
        self.sha_proj = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.spe_proj = nn.Linear(self.output_dim, self.output_dim, bias=False)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        h_sha = self.backbone(x)
        residual_input = torch.concat([h_sha.detach(), x], dim=-1)
        residuals = torch.stack([expert(residual_input) for expert in self.residual_experts], dim=1)
        h_spe = domain_select(residuals, domain_ids)
        self._cos_sim = F.cosine_similarity(h_sha, h_spe, dim=-1)
        return self.sha_proj(h_sha) + self.spe_proj(h_spe)

    def aux_loss(self) -> torch.Tensor:
        if self._cos_sim is None:
            return torch.tensor(0.0, device=self.sha_proj.weight.device)
        return torch.clamp(self._cos_sim.abs() - self.margin, min=0.0).mean()


class RIPLEPLELayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            detach_gate_input: bool = True,
            margin: float = 0.1,
            aux_loss_weight: float = 0.1,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.num_specific_experts = int(num_specific_experts)
        self.output_dim = int(expert_dims[-1])
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
                for _ in range(self.num_specific_experts)
            ])
            for _ in range(self.num_domains)
        ])
        self.specific_gates = nn.ModuleList([
            nn.Sequential(
                build_mlp(
                    input_dim,
                    (max(1, input_dim // 2), self.num_specific_experts + int(num_shared_experts)),
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                ),
                nn.Softmax(dim=-1),
            )
            for _ in range(self.num_domains)
        ])
        self.shared_experts = nn.ModuleList([
            RIPLESharedExpert(
                input_dim=input_dim,
                num_domains=num_domains,
                hidden_dims=expert_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
                margin=margin,
                aux_loss_weight=aux_loss_weight,
            )
            for _ in range(int(num_shared_experts))
        ])
    def forward(self, task_inputs: list[torch.Tensor], shared_input: torch.Tensor, domain_ids: torch.Tensor):
        shared_outputs = [expert(shared_input, domain_ids=domain_ids) for expert in self.shared_experts]
        shared_state = torch.stack(shared_outputs, dim=1).mean(dim=1)

        task_outputs = []
        for domain_idx in range(self.num_domains):
            task_input = task_inputs[domain_idx]
            gate_input = task_input.detach() if self.detach_gate_input else task_input
            specific_outputs = [expert(task_input) for expert in self.specific_experts[domain_idx]]
            mix_inputs = torch.stack(specific_outputs + shared_outputs, dim=1)
            gate = self.specific_gates[domain_idx](gate_input).unsqueeze(-1)
            task_outputs.append(torch.sum(gate * mix_inputs, dim=1))
        return task_outputs, shared_state

    def aux_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        return sum((expert.aux_loss() for expert in self.shared_experts), torch.tensor(0.0, device=device))


class RIPLEEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            detach_gate_input: bool = True,
            margin: float = 0.1,
            aux_loss_weight: float = 0.5
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.output_dim = int(expert_dims[-1])
        self.layers = nn.ModuleList()

        layer_input_dim = int(input_dim)
        for _ in range(int(num_levels)):
            layer = RIPLEPLELayer(
                input_dim=layer_input_dim,
                num_domains=num_domains,
                expert_dims=expert_dims,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
                detach_gate_input=detach_gate_input,
                margin=margin,
                aux_loss_weight=aux_loss_weight,
            )
            self.layers.append(layer)
            layer_input_dim = layer.output_dim

        self._latest_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        task_inputs = [x for _ in range(self.num_domains)]
        shared_input = x
        for layer in self.layers:
            task_inputs, shared_input = layer(task_inputs, shared_input, domain_ids)
        self._latest_aux_loss = sum((layer.aux_loss() for layer in self.layers), torch.tensor(0.0, device=x.device))
        return select_domain_output(task_outputs=task_inputs, domain_ids=domain_ids)

    def aux_loss(self) -> torch.Tensor:
        if self._latest_aux_loss is None:
            return torch.tensor(0.0)
        return self._latest_aux_loss


class RIPLEModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            detach_gate_input: bool = True,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            margin: float = 0.1,
            aux_loss_weight: float = 0.5,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        self.encoder = RIPLEEncoder(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_levels=num_levels,
            num_specific_experts=num_specific_experts,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            detach_gate_input=detach_gate_input,
            margin=margin,
            aux_loss_weight=aux_loss_weight,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.encoder.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, domain_ids), domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        aux_loss = self.encoder.aux_loss()
        return bce_loss + aux_loss


if __name__ == '__main__':
    from betterbole.models.utils.tests import DummyCls

    riple = RIPLEModel(DummyCls.MANAGER, DummyCls.NUM_DOMAINS)
    loss = riple.calculate_loss(DummyCls.make_interaction())
    print(loss)
