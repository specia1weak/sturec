from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from betterbole.models.msr.automtl.modules import GateFunc


class MixedExpert(nn.Module):
    def __init__(self, gate_input_dim: int, num_choices: int):
        super().__init__()
        self.num_choices = int(num_choices)
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, self.num_choices, bias=False),
            nn.Softmax(dim=1),
        )
        self.beta = nn.Parameter(torch.zeros(self.num_choices))
        self.selection_gate = GateFunc.apply
        self.in_warmup = True

    @property
    def module_str(self) -> str:
        probs = torch.sigmoid(self.beta).detach().cpu().numpy()
        return f"Mixed expert select probs: {probs}"

    def set_chosen_op_active(self) -> None:
        self.in_warmup = False

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        expert_selection = self.selection_gate(torch.sigmoid(self.beta), self.in_warmup)
        expert_outputs = xs[:-1]
        gate_input = xs[-1]
        gate_value = self.gate_network(gate_input)
        output = 0
        for idx in range(self.num_choices):
            output = output + gate_value[:, idx].unsqueeze(1) * expert_outputs[idx] * expert_selection[idx]
        return output

    def export_arch(self, beta: Optional[list[float]] = None) -> list[float]:
        if beta is None:
            self.beta.data = (torch.sigmoid(self.beta.data) > 0.5).float()
        else:
            self.beta.data = torch.tensor(beta, device=self.beta.device)
        self.beta.requires_grad = False
        self.in_warmup = False
        return self.beta.tolist()
