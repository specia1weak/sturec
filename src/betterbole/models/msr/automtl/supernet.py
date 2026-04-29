import copy
from typing import Iterable, Optional

import torch
import torch.nn as nn

from betterbole.models.msr.automtl.mix_expert import MixedExpert
from betterbole.models.msr.automtl.mix_feature import MixFeature
from betterbole.models.msr.automtl.mix_op import MixedOp
from betterbole.models.msr.automtl.modules import FM, build_candidate_ops, init_linear_layers
from betterbole.models.utils.container import domain_select


class ExpertModule(nn.Module):
    def __init__(
            self,
            input_dim: int,
            in_features,
            out_features,
            num_layers: int,
            candidate_ops: Iterable[str],
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        if isinstance(in_features, (list, tuple)):
            self.in_features = [int(input_dim)] + [int(v) for v in in_features]
        else:
            self.in_features = [int(input_dim)] + [int(in_features)] * (int(num_layers) - 1)

        if isinstance(out_features, (list, tuple)):
            self.out_features = [int(v) for v in out_features]
        else:
            self.out_features = [int(out_features)] * int(num_layers)

        self.blocks = nn.ModuleList([
            MixedOp(
                build_candidate_ops(
                    candidate_ops=candidate_ops,
                    input_dim=self.in_features[layer_idx],
                    output_dim=self.out_features[layer_idx],
                    dropout_rate=dropout_rate,
                )
            )
            for layer_idx in range(int(num_layers))
        ])

    @property
    def module_str(self) -> str:
        return "Expert module: " + " | ".join(block.module_str for block in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class AutoMTLSuperNet(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_sparse_fields: int,
            num_dense_fields: int,
            num_domains: int,
            num_experts: int,
            num_expert_layers: int,
            expert_num_layers: int,
            expert_in_features,
            expert_out_features,
            dropout_rate: float,
            expert_candidate_ops: Iterable[str],
    ):
        self.embedding_dim = int(embedding_dim)
        self.num_sparse_fields = int(num_sparse_fields)
        self.num_dense_fields = int(num_dense_fields)
        self.gate_input_dim = self.embedding_dim * (self.num_sparse_fields + 1) + self.num_dense_fields
        if isinstance(expert_out_features, (list, tuple)):
            self.output_dim = int(expert_out_features[-1])
        else:
            self.output_dim = int(expert_out_features)
        super().__init__()
        self.input_dim = self.gate_input_dim
        self.num_domains = int(num_domains)
        self.num_experts = int(num_experts)
        self.num_expert_layers = int(num_expert_layers)

        self._exported_arch = {}
        self._selection_activated = False

        self.interaction_layer = FM()
        self.feature_modules = nn.ModuleList([
            MixFeature(self.num_sparse_fields, self.interaction_layer)
            for _ in range(self.num_experts)
        ])
        expert_input_dims = [self.gate_input_dim] + [self.output_dim] * (self.num_expert_layers - 1)
        self.experts = nn.ModuleList([
            nn.ModuleList([
                ExpertModule(
                    input_dim=input_dim,
                    in_features=expert_in_features,
                    out_features=expert_out_features,
                    num_layers=expert_num_layers,
                    candidate_ops=expert_candidate_ops,
                    dropout_rate=dropout_rate,
                )
                for _ in range(self.num_experts)
            ])
            for input_dim in expert_input_dims
        ])
        self.mixed_experts = nn.ModuleList([
            nn.ModuleList([
                MixedExpert(self.gate_input_dim, self.num_experts)
                for _ in range(self.num_experts if layer_idx < self.num_expert_layers - 1 else self.num_domains)
            ])
            for layer_idx in range(self.num_expert_layers)
        ])
        init_linear_layers(self)

    @property
    def exported_arch(self) -> dict:
        return self._exported_arch

    @property
    def redundant_modules(self) -> list[nn.Module]:
        modules = []
        for module in self.modules():
            if isinstance(module, (MixFeature, MixedExpert)):
                modules.append(module)
        return modules

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if "alpha" in name or "beta" in name:
                yield param

    def alpha_parameters(self):
        for name, param in self.named_parameters():
            if "alpha" in name:
                yield param

    def beta_parameters(self):
        for name, param in self.named_parameters():
            if "beta" in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if "alpha" not in name and "beta" not in name:
                yield param

    def init_arch_params(self, init_type: str = "normal", init_ratio: float = 1e-3) -> None:
        for param in self.alpha_parameters():
            if init_type == "normal":
                param.data.normal_(0, init_ratio)
            elif init_type == "uniform":
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError(f"Unsupported AutoMTL arch init: {init_type}")
        for param in self.beta_parameters():
            param.data.zero_()

    def set_chosen_op_active(self) -> None:
        if self._selection_activated:
            return
        for module in self.redundant_modules:
            module.set_chosen_op_active()
        self._selection_activated = True

    def discretize_one_op(self) -> Optional[str]:
        mix_ops = []
        for name, module in self.named_modules():
            if isinstance(module, MixedOp):
                mix_ops.append((name, module, module.entropy()))
        if not mix_ops:
            return None
        op_name, discretize_op, _ = min(mix_ops, key=lambda item: item[2])
        tokens = op_name.split(".")
        parent_name = ".".join(tokens[:-1])
        parent = self.get_submodule(parent_name)
        idx, op = discretize_op.discretize()
        self._exported_arch[op_name] = idx
        parent.add_module(tokens[-1], op)
        return op_name

    def export_architecture(self) -> dict:
        mix_ops = []
        for name, module in self.named_modules():
            if isinstance(module, MixedOp):
                mix_ops.append((name, module))

        for op_name, discretize_op in mix_ops:
            tokens = op_name.split(".")
            parent_name = ".".join(tokens[:-1])
            parent = self.get_submodule(parent_name)
            idx, op = discretize_op.discretize()
            self._exported_arch[op_name] = idx
            parent.add_module(tokens[-1], op)

        for name, module in self.named_modules():
            if isinstance(module, MixFeature):
                self._exported_arch[name] = module.export_arch()
            elif isinstance(module, MixedExpert):
                self._exported_arch[name] = module.export_arch()
        return copy.deepcopy(self._exported_arch)

    def convert_to_normal_net(self, arch_config: dict) -> None:
        for name, module in self.named_modules():
            if isinstance(module, MixedOp):
                op = module.discretize(chosen_idx=arch_config[name])
                tokens = name.split(".")
                parent_name = ".".join(tokens[:-1])
                parent = self.get_submodule(parent_name)
                parent.add_module(tokens[-1], op)
            elif isinstance(module, MixFeature):
                module.export_arch(*arch_config[name])
            elif isinstance(module, MixedExpert):
                module.export_arch(arch_config[name])
        self._exported_arch = copy.deepcopy(arch_config)

    def describe_architecture(self) -> list[str]:
        lines = []
        for idx, feature_module in enumerate(self.feature_modules):
            lines.append(f"Mixed Feature {idx}: {feature_module.module_str}")
        for layer_idx, experts in enumerate(self.experts):
            for expert_idx, expert in enumerate(experts):
                lines.append(f"Expert {layer_idx}-{expert_idx}: {expert.module_str}")
        for layer_idx, mixers in enumerate(self.mixed_experts):
            for mix_idx, mix in enumerate(mixers):
                lines.append(f"Mixed Expert {layer_idx}-{mix_idx}: {mix.module_str}")
        return lines

    def forward(
            self,
            sparse_embs: torch.Tensor,
            dense_features: Optional[torch.Tensor],
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        mix_features = [
            feature_module(sparse_embs, dense_features)
            for feature_module in self.feature_modules
        ]
        gate_input = torch.nn.functional.pad(
            sparse_embs.flatten(start_dim=1),
            (0, self.embedding_dim + self.num_dense_fields),
        )
        mix_features.append(gate_input)

        temp = []
        for layer_idx in range(self.num_expert_layers - 1):
            for expert_idx in range(self.num_experts):
                mix_features[expert_idx] = self.experts[layer_idx][expert_idx](mix_features[expert_idx])
            for mix_idx in range(self.num_experts):
                temp.append(self.mixed_experts[layer_idx][mix_idx](mix_features))
            temp.append(mix_features[-1])
            mix_features = temp
            temp = []

        for expert_idx in range(self.num_experts):
            mix_features[expert_idx] = self.experts[-1][expert_idx](mix_features[expert_idx])
        for domain_idx in range(self.num_domains):
            temp.append(self.mixed_experts[-1][domain_idx](mix_features))
        stacked = torch.stack(temp, dim=1)
        return domain_select(stacked, domain_ids.long())
