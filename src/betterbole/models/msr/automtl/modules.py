import math
from typing import Iterable

import torch
import torch.nn as nn


class GateFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, in_warmup: bool):
        if in_warmup:
            return torch.ones_like(inputs)
        if torch.is_grad_enabled():
            return torch.bernoulli(inputs)
        return (inputs > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class FM(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sum_squared = torch.pow(torch.sum(x, dim=1), 2)
        squared_sum = torch.sum(torch.pow(x, 2), dim=1)
        return 0.5 * (sum_squared - squared_sum)


class IdentityOp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection = None
        if self.input_dim != self.output_dim:
            self.projection = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection is not None:
            return self.projection(x)
        return x

    @property
    def module_str(self) -> str:
        if self.projection is None:
            return "Identity"
        return f"{self.input_dim}->{self.output_dim}_Project"

    @staticmethod
    def is_zero_layer() -> bool:
        return False


class ZeroOp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = list(x.shape)
        output_shape[-1] = self.output_dim
        return x.new_zeros(output_shape)

    @property
    def module_str(self) -> str:
        return "Zero"

    @staticmethod
    def is_zero_layer() -> bool:
        return True


class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.hidden = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.projection = nn.Linear(self.hidden_size, self.output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.projection(x)

    @property
    def module_str(self) -> str:
        return f"{self.input_dim}->{self.hidden_size}->{self.output_dim}_MLP"

    @staticmethod
    def is_zero_layer() -> bool:
        return False


def build_candidate_ops(
        candidate_ops: Iterable[str],
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
) -> list[nn.Module]:
    name_to_op = {
        "Identity": lambda: IdentityOp(input_dim, output_dim),
        "Zero": lambda: ZeroOp(input_dim, output_dim),
        "MLP-16": lambda: CandidateMLP(input_dim, output_dim, 16, dropout_rate),
        "MLP-32": lambda: CandidateMLP(input_dim, output_dim, 32, dropout_rate),
        "MLP-64": lambda: CandidateMLP(input_dim, output_dim, 64, dropout_rate),
        "MLP-128": lambda: CandidateMLP(input_dim, output_dim, 128, dropout_rate),
        "MLP-256": lambda: CandidateMLP(input_dim, output_dim, 256, dropout_rate),
        "MLP-512": lambda: CandidateMLP(input_dim, output_dim, 512, dropout_rate),
        "MLP-1024": lambda: CandidateMLP(input_dim, output_dim, 1024, dropout_rate),
    }
    ops = []
    for name in candidate_ops:
        if name not in name_to_op:
            raise ValueError(f"Unsupported AutoMTL candidate op: {name}")
        ops.append(name_to_op[name]())
    return ops


def init_linear_layers(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            stdv = 1.0 / math.sqrt(submodule.weight.size(1))
            submodule.weight.data.uniform_(-stdv, stdv)
            if submodule.bias is not None:
                submodule.bias.data.zero_()
        elif isinstance(submodule, nn.BatchNorm1d):
            submodule.weight.data.fill_(1.0)
            submodule.bias.data.zero_()
