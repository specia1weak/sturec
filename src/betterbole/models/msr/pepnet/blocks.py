from __future__ import annotations

import torch
from torch import nn


class PPBlock(nn.Module):
    def __init__(
            self,
            agnostic_dim: int,
            gate_input_dim: int,
            hidden_dims,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        dims = (agnostic_dim, *hidden_dims)
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.gates.append(nn.Sequential(nn.Linear(gate_input_dim, out_dim), nn.Sigmoid()))
            self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, agnostic_x: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        hidden = agnostic_x
        last_layer_idx = len(self.layers) - 1
        for layer_idx, (layer, gate, dropout) in enumerate(zip(self.layers, self.gates, self.dropouts)):
            hidden = layer(hidden)
            hidden = hidden * gate(gate_input)
            if layer_idx != last_layer_idx:
                hidden = torch.relu(hidden)
                hidden = dropout(hidden)
        return hidden
