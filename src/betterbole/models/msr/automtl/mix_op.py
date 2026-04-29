import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedOp(nn.Module):
    def __init__(self, candidate_ops: list[nn.Module]):
        super().__init__()
        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.alpha = nn.Parameter(torch.zeros(self.n_choices))

    @property
    def n_choices(self) -> int:
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self) -> torch.Tensor:
        return F.softmax(self.alpha, dim=0)

    @property
    def chosen_index(self) -> tuple[int, float]:
        probs = self.probs_over_ops.detach().cpu().numpy()
        index = int(np.argmax(probs))
        return index, float(probs[index])

    @property
    def chosen_op(self) -> nn.Module:
        return self.candidate_ops[self.chosen_index[0]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op_results = torch.stack([op(x) for op in self.candidate_ops], dim=0)
        alpha_shape = (-1,) + (1,) * (op_results.dim() - 1)
        return torch.sum(op_results * self.probs_over_ops.view(*alpha_shape), dim=0)

    @property
    def module_str(self) -> str:
        index, prob = self.chosen_index
        chosen_op = self.candidate_ops[index]
        op_name = getattr(chosen_op, "module_str", chosen_op.__class__.__name__)
        return f"MixOp({op_name}, {prob:.3f})"

    def entropy(self, eps: float = 1e-8) -> float:
        probs = self.probs_over_ops
        return float(-(probs * torch.log(probs + eps)).sum().item())

    def discretize(self, chosen_idx: int = None):
        if chosen_idx is not None:
            return self.candidate_ops[int(chosen_idx)]
        idx, _ = self.chosen_index
        return idx, self.candidate_ops[idx]
