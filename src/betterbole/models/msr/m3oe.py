from typing import Dict, Iterable, Optional, Type

import torch
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.core.train.hooks import CustomTrainStepProtocol, TrainerHooksProtocol
from betterbole.core.interaction import Interaction
from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.container import domain_select


class BalanceFactor(nn.Module):
    def __init__(self, initial_value: float = 1.0):
        super().__init__()
        self.balance_factor = nn.Parameter(torch.tensor(float(initial_value)))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.balance_factor)


class DomainBalanceFactor(nn.Module):
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
    def __init__(self, dims):
        super().__init__()
        dims = to_dims(dims, ())
        if len(dims) < 2:
            raise ValueError("LayerNormMLP requires at least input and output dims")
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class M3oEBackbone(nn.Module):
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

        super().__init__()
        self.output_dim = int(expert_dims[-1])
        self.star_dims = star_dims
        self.expert_dims = expert_dims
        self.num_domains = int(num_domains)
        self.num_shared_experts = int(num_shared_experts)
        self.factor_mode = (factor_mode or "global").lower()
        self.shared_gate_detach = True if shared_gate_detach is None else bool(shared_gate_detach)

        star_hidden_dim, star_output_dim = star_dims
        expert_mlp_dims = (star_output_dim, *expert_dims)

        self.skip_conn = LayerNormMLP((input_dim, star_output_dim))
        self.shared_weight = nn.Parameter(torch.empty(input_dim, star_hidden_dim))
        self.shared_bias = nn.Parameter(torch.zeros(star_hidden_dim))
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(input_dim, star_hidden_dim)) for _ in range(num_domains)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(star_hidden_dim)) for _ in range(num_domains)])
        self.star_mlp = LayerNormMLP((star_hidden_dim, star_output_dim))

        nn.init.xavier_uniform_(self.shared_weight.data)
        for slot_weight in self.slot_weight:
            nn.init.xavier_uniform_(slot_weight.data)

        self.shared_experts = nn.ModuleList([LayerNormMLP(expert_mlp_dims) for _ in range(self.num_shared_experts)])
        self.domain_experts = nn.ModuleList([LayerNormMLP(expert_mlp_dims) for _ in range(num_domains)])
        self.shared_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(star_output_dim, self.num_shared_experts), nn.Softmax(dim=-1))
            for _ in range(num_domains)
        ])

        if self.factor_mode == "dynamic":
            self.balance_gate = nn.Sequential(nn.Linear(star_output_dim, 2), nn.Softmax(dim=-1))
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
            return balance[:, :1] * shared_state + balance[:, 1:] * domain_state

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
        if isinstance(self.domain_expert_factor, DomainBalanceFactor):
            domain_expert_factor = self.domain_expert_factor(domain_ids)
        else:
            domain_expert_factor = self.domain_expert_factor()
        self.last_outer_factor = domain_expert_factor.detach()
        return shared_state + domain_expert_factor * domain_state

    def arch_parameters_name(self):
        return ("balance_factor", "balance_gate")

    def arch_parameters(self):
        params = []
        balance_gate = getattr(self, "balance_gate", None)
        if balance_gate is not None:
            params.extend(list(balance_gate.parameters()))
        for module_name in ("domain_expert_factor", "domain_balance_factor"):
            module = getattr(self, module_name, None)
            if module is not None:
                params.extend(list(module.parameters()))
        return params


class M3oEVersion1Backbone(M3oEBackbone):
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


class _BaseM3oEModel(MSRModel, CustomTrainStepProtocol, TrainerHooksProtocol):
    BACKBONE_CLS: Type[M3oEBackbone] = M3oEBackbone

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            star_dims: Iterable[int] = (512, 256),
            expert_dims: Iterable[int] = (64,),
            num_shared_experts: int = 4,
            factor_mode: str = None,
            shared_gate_detach: bool = None,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            arch_lr: float = 1e-3,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        backbone_kwargs = dict(
            input_dim=self.input_dim,
            num_domains=num_domains,
            star_dims=star_dims,
            expert_dims=expert_dims,
            num_shared_experts=num_shared_experts,
        )
        if self.BACKBONE_CLS is M3oEBackbone:
            backbone_kwargs["factor_mode"] = factor_mode
            backbone_kwargs["shared_gate_detach"] = shared_gate_detach
        self.backbone = self.BACKBONE_CLS(**backbone_kwargs)
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )
        self.arch_lr = float(arch_lr)
        self.arch_optimizer: Optional[torch.optim.Optimizer] = None
        self._cached_arch_batch: Optional[Interaction] = None
        self._latest_arch_debug: Dict[str, float] = {}

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

    def _arch_parameters(self):
        return [param for param in self.backbone.arch_parameters() if param.requires_grad]

    def _ensure_arch_optimizer(self, weight_decay: float = 0.0) -> Optional[torch.optim.Optimizer]:
        arch_params = self._arch_parameters()
        if not arch_params:
            return None
        if self.arch_optimizer is None:
            self.arch_optimizer = torch.optim.Adam(
                arch_params,
                lr=self.arch_lr,
                weight_decay=float(weight_decay),
            )
        return self.arch_optimizer

    def _clear_arch_grads(self) -> None:
        for param in self._arch_parameters():
            param.grad = None

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        self._clear_arch_grads()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        self._cached_arch_batch = batch_interaction.cpu()
        return float(loss.item())

    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        del ctx
        self._cached_arch_batch = None

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        arch_optimizer = self._ensure_arch_optimizer(
            weight_decay=ctx.optimizer.param_groups[0].get("weight_decay", 0.0)
        )
        if arch_optimizer is None or self._cached_arch_batch is None:
            return

        arch_batch = self._cached_arch_batch.to(ctx.cfg.device)
        self.zero_grad(set_to_none=True)
        arch_optimizer.zero_grad(set_to_none=True)
        loss = self.calculate_loss(arch_batch)
        loss.backward()
        arch_optimizer.step()
        self.zero_grad(set_to_none=True)

        debug = {
            "arch_loss": float(loss.item()),
        }
        if self.backbone.last_inner_factor is not None:
            debug["inner_factor"] = float(self.backbone.last_inner_factor.float().mean().item())
        if self.backbone.last_outer_factor is not None:
            debug["outer_factor"] = float(self.backbone.last_outer_factor.float().mean().item())
        self._latest_arch_debug = debug
        print(
            "[M3oE Arch] "
            + " ".join(f"{key}={value:.4f}" for key, value in debug.items())
        )

    def on_eval_epoch_end(self, metrics: Optional[dict], ctx: TrainContext) -> None:
        del metrics, ctx
        return


class M3oEModel(_BaseM3oEModel):
    BACKBONE_CLS = M3oEBackbone


class M3oEVersion1Model(_BaseM3oEModel):
    BACKBONE_CLS = M3oEVersion1Backbone


class M3oEVersion2Model(_BaseM3oEModel):
    BACKBONE_CLS = M3oEVersion2Backbone
