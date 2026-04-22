import torch
from torch import nn

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import build_mlp, to_dims
from betterbole.models.utils.container import domain_select
"""
使用M3oE请每隔1000batch单独更新一次arch_params
"""

class M3oEBackbone(MSRBackbone):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_shared_experts: int = 3,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=expert_dims[-1])
        self.num_shared_experts = int(num_shared_experts)
        self.star_shared = nn.Linear(input_dim, expert_dims[0])
        self.star_domain = nn.ModuleList([
            nn.Linear(input_dim, expert_dims[0]) for _ in range(num_domains)
        ])
        self.star_project = build_mlp(
            expert_dims[0],
            expert_dims[1:] if len(expert_dims) > 1 else (expert_dims[0],),
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.shared_experts = nn.ModuleList([
            build_mlp(
                expert_dims[-1],
                (expert_dims[-1],),
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_shared_experts)
        ])
        self.domain_experts = nn.ModuleList([
            build_mlp(
                expert_dims[-1],
                (expert_dims[-1],),
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(num_domains)
        ])
        self.shared_gate = nn.Sequential(
            nn.Linear(expert_dims[-1], self.num_shared_experts),
            nn.Softmax(dim=-1),
        )
        self.balance_gate = nn.Sequential(
            nn.Linear(expert_dims[-1], 2),
            nn.Softmax(dim=-1),
        )
        self.output_project = build_mlp(
            expert_dims[-1],
            (expert_dims[-1],),
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=False,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        star_outputs = []
        for domain_layer in self.star_domain:
            star_outputs.append(torch.relu(self.star_shared(x) * domain_layer(x)))
        star_state = domain_select(torch.stack(star_outputs, dim=1), domain_ids.long())
        star_state = self.star_project(star_state)

        shared_outputs = torch.stack([expert(star_state) for expert in self.shared_experts], dim=1)
        shared_gate = self.shared_gate(star_state).unsqueeze(-1)
        shared_state = torch.sum(shared_gate * shared_outputs, dim=1)

        domain_outputs = torch.stack([expert(star_state) for expert in self.domain_experts], dim=1)
        domain_state = domain_select(domain_outputs, domain_ids.long())

        balance = self.balance_gate(star_state)
        mixed = balance[:, :1] * shared_state + balance[:, 1:] * domain_state
        return self.output_project(mixed + star_state)

    def get_parameter_groups(self):
        """
        将参数分为架构参数(Architecture)和普通参数(Weights)
        注意：现在返回的是 (name, param) 的元组列表
        """
        arch_named_params = []
        base_named_params = []
        for name, param in self.named_parameters():
            if 'gate' in name or 'balance' in name:
                arch_named_params.append((name, param))
            else:
                base_named_params.append((name, param))
        return arch_named_params, base_named_params


def create_m3oe(
        input_dim: int = 48,
        num_domains: int = 3,
        batch_size: int = 8,
) -> tuple[M3oEBackbone, torch.Tensor]:
    from betterbole.models.utils.tests import dummy_input_multi_domain

    model = M3oEBackbone(input_dim=input_dim, num_domains=num_domains)
    x, domain_ids = dummy_input_multi_domain(num_domains=num_domains, batch_size=batch_size, emb_size=input_dim)
    out = model(x, domain_ids)
    assert out.shape == (batch_size, model.output_dim)
    return model, out


if __name__ == '__main__':
    m3oe, _ = create_m3oe()
    arch_params, base_params = m3oe.get_parameter_groups()
    optimizer_base = torch.optim.Adam(
        base_params,
        lr=1e-3,
        weight_decay=1e-5
    )
    optimizer_arch = torch.optim.Adam(
        arch_params,
        lr=1e-4,
        weight_decay=0
    )
    from betterbole.models.utils.tests import dummy_input_multi_domain
    for epoch in range(10):
        optimizer_base.zero_grad()
        for batch in range(30):
            x, domain_ids = dummy_input_multi_domain(num_domains=3, batch_size=10, emb_size=48)
            out = m3oe(x, domain_ids)

            loss = (out - 1)
            optimizer_base.zero_grad()
            loss.backward()
            optimizer_base.step()


        optimizer_arch.step()
