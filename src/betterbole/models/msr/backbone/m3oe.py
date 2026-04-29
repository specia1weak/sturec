import torch
from torch import nn

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import to_dims
from betterbole.models.utils.container import domain_select
"""
使用M3oE时可以将 balance_factor 参数单独低频更新，其他权重正常更新。
"""


class BalanceFactor(nn.Module):
    """
    原版 M3oE 中 _weight_* 风格的全局可学习融合因子。

    参数本身不依赖样本输入，forward 后通过 sigmoid 约束到 (0, 1)。
    """

    def __init__(self, initial_value: float = 1.0):
        super().__init__()
        self.balance_factor = nn.Parameter(torch.tensor(float(initial_value)))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.balance_factor)


class DomainBalanceFactor(nn.Module):
    """
    只以 domain_id 为输入的可学习融合因子。

    每个 domain 拥有一个独立标量，forward 后输出 batch 级别的 [B, 1] 因子。
    """

    def __init__(self, num_domains: int, initial_value: float = 1.0):
        super().__init__()
        self.num_domains = int(num_domains)
        self.balance_factor = nn.Embedding(num_domains, 1)
        nn.init.constant_(self.balance_factor.weight, float(initial_value))

    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        domain_ids = domain_ids.long()
        if domain_ids.numel() > 0 and (domain_ids.min() < 0 or domain_ids.max() >= self.num_domains):
            raise ValueError(
                f"domain_ids out of range: min={domain_ids.min().item()}, "
                f"max={domain_ids.max().item()}, num_domains={self.num_domains}"
            )
        return torch.sigmoid(self.balance_factor(domain_ids))


class LayerNormMLP(nn.Module):
    """
    与参考 M3oE/MDMTRec 中 MLP_N 对齐的 MLP。

    每一层都是 Linear -> LayerNorm -> ReLU，包括最后一层。
    """

    def __init__(self, dims):
        super().__init__()
        dims = to_dims(dims, ())
        if len(dims) < 2:
            raise ValueError("LayerNormMLP requires at least input and output dims")

        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class M3oEBackbone(MSRBackbone):
    """
    M3oE 的单任务多场景主干，按参考源码的 domain 相关路径实现。

    参数语义：
    - star_dims: STAR 层维度，不包含 input_dim，默认 (512, 256)，即 input_dim -> 512 -> 256。
    - expert_dims: shared/domain expert 的输出维度，不包含 STAR 输出维度，默认 (64,)，即 256 -> 64。
    - num_shared_experts: 共享 expert 数量，参考源码默认 expert_num=4。

    注意：这里不包含任务级 expert；输出是融合后的场景表征，外部再接 domain tower。
    """

    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            star_dims=None,
            expert_dims=None,
            num_shared_experts: int = 4,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            factor_mode: str = None,
            shared_gate_detach: bool = None,
    ):
        if dropout_rate != 0.0 or activation != "relu" or batch_norm:
            raise ValueError(
                "M3oEBackbone follows the reference MLP_N structure "
                "(Linear -> LayerNorm -> ReLU) and does not use dropout_rate/activation/batch_norm."
            )
        star_dims = to_dims(star_dims, (512, 256))
        if len(star_dims) != 2:
            raise ValueError("star_dims must be (star_hidden_dim, star_output_dim), e.g. (512, 256)")
        expert_dims = to_dims(expert_dims, (64,))
        if len(expert_dims) < 1:
            raise ValueError("expert_dims must contain at least one output dimension, e.g. (64,)")

        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=expert_dims[-1])
        self.star_dims = star_dims
        self.expert_dims = expert_dims
        self.num_shared_experts = int(num_shared_experts)
        self.factor_mode = (factor_mode or "global").lower()
        self.shared_gate_detach = True if shared_gate_detach is None else bool(shared_gate_detach)

        star_hidden_dim, star_output_dim = star_dims
        expert_mlp_dims = (star_output_dim, *expert_dims)

        self.skip_conn = LayerNormMLP((input_dim, star_output_dim))
        self.shared_weight = nn.Parameter(torch.empty(input_dim, star_hidden_dim))
        self.shared_bias = nn.Parameter(torch.zeros(star_hidden_dim))
        self.slot_weight = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, star_hidden_dim))
            for _ in range(num_domains)
        ])
        self.slot_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(star_hidden_dim))
            for _ in range(num_domains)
        ])
        self.star_mlp = LayerNormMLP((star_hidden_dim, star_output_dim))

        nn.init.xavier_uniform_(self.shared_weight.data)
        for slot_weight in self.slot_weight:
            nn.init.xavier_uniform_(slot_weight.data)

        self.shared_experts = nn.ModuleList([
            LayerNormMLP(expert_mlp_dims)
            for _ in range(self.num_shared_experts)
        ])
        self.domain_experts = nn.ModuleList([
            LayerNormMLP(expert_mlp_dims)
            for _ in range(num_domains)
        ])
        self.shared_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(star_output_dim, self.num_shared_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_domains)
        ])
        if self.factor_mode == "dynamic":
            self.balance_gate = nn.Sequential(
                nn.Linear(star_output_dim, 2),
                nn.Softmax(dim=-1),
            )
        elif self.factor_mode == "domain":
            self.domain_expert_factor = DomainBalanceFactor(num_domains, initial_value=1.0)
            self.domain_balance_factor = DomainBalanceFactor(num_domains, initial_value=1.0)
        elif self.factor_mode in ("beta_domain", "v2"):
            self.domain_expert_factor = BalanceFactor(initial_value=1.0)
            self.domain_balance_factor = DomainBalanceFactor(num_domains, initial_value=1.0)
        elif self.factor_mode == "global":
            self.domain_expert_factor = BalanceFactor(initial_value=1.0)
            self.domain_balance_factor = BalanceFactor(initial_value=1.0)
        else:
            raise ValueError(
                "Unsupported M3oE factor_mode: "
                f"{self.factor_mode}. Choose from global, dynamic, domain, beta_domain."
            )

        self.last_inner_factor = None
        self.last_outer_factor = None

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        domain_ids = domain_ids.long()
        star_outputs = []
        for slot_weight, slot_bias in zip(self.slot_weight, self.slot_bias):
            slot_output = torch.matmul(x, torch.multiply(slot_weight, self.shared_weight))
            star_outputs.append(slot_output + slot_bias + self.shared_bias)
        star_state = domain_select(torch.stack(star_outputs, dim=1), domain_ids)
        star_state = self.star_mlp(star_state) + self.skip_conn(x)

        shared_outputs = torch.stack([expert(star_state) for expert in self.shared_experts], dim=1)
        gate_input = star_state.detach() if self.shared_gate_detach else star_state
        gate_outputs = torch.stack([gate(gate_input) for gate in self.shared_gates], dim=1)
        shared_gate = domain_select(gate_outputs, domain_ids).unsqueeze(1)
        shared_state = torch.bmm(shared_gate, shared_outputs).squeeze(1)

        domain_outputs = torch.stack([expert(star_state) for expert in self.domain_experts], dim=1)
        if self.factor_mode == "dynamic":
            domain_state = domain_select(domain_outputs, domain_ids)
            balance = self.balance_gate(star_state)
            mixed = balance[:, :1] * shared_state + balance[:, 1:] * domain_state
            return mixed

        if self.num_domains > 1:
            if isinstance(self.domain_balance_factor, DomainBalanceFactor):
                own_domain_weight = self.domain_balance_factor(domain_ids).unsqueeze(-1)
            else:
                own_domain_weight = self.domain_balance_factor()
            self.last_inner_factor = own_domain_weight.detach()
            other_domain_weight = (1.0 - own_domain_weight) / (self.num_domains - 1)
            all_domain_state = domain_outputs.sum(dim=1, keepdim=True)
            balanced_domain_outputs = (
                own_domain_weight * domain_outputs
                + other_domain_weight * (all_domain_state - domain_outputs)
            )
        else:
            balanced_domain_outputs = domain_outputs

        domain_state = domain_select(balanced_domain_outputs, domain_ids)
        domain_expert_factor = self.domain_expert_factor()
        if isinstance(self.domain_expert_factor, DomainBalanceFactor):
            domain_expert_factor = self.domain_expert_factor(domain_ids)
        self.last_outer_factor = domain_expert_factor.detach()
        return shared_state + domain_expert_factor * domain_state

    def arch_parameters_name(self):
        return ('balance_factor', 'balance_gate')


class M3oEVersion1Backbone(M3oEBackbone):
    """
    M3oE 的 domain-conditioned factor 版本。

    与 M3oEBackbone 的区别：
    - 原版 BalanceFactor 是全局标量，不看输入。
    - Version1 的 DomainBalanceFactor 只看 domain_id，每个 domain 学自己的标量。
    """

    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            star_dims=None,
            expert_dims=None,
            num_shared_experts: int = 4,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__(
            input_dim=input_dim,
            num_domains=num_domains,
            star_dims=star_dims,
            expert_dims=expert_dims,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            factor_mode="domain",
        )


class M3oEVersion2Backbone(M3oEBackbone):
    """
    M3oE 的 beta-only domain-conditioned 版本。

    设计：
    - alpha = domain_expert_factor 仍然是全局标量
    - beta = domain_balance_factor 改为只看 domain_id 的 per-domain 标量
    """

    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            star_dims=None,
            expert_dims=None,
            num_shared_experts: int = 4,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__(
            input_dim=input_dim,
            num_domains=num_domains,
            star_dims=star_dims,
            expert_dims=expert_dims,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            factor_mode="beta_domain",
        )


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
    arch_params, base_params = m3oe.arch_parameters()
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
