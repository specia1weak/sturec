import torch
from torch import nn

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import to_dims
from betterbole.models.utils.container import domain_select


class _PPBlock(nn.Module):
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


class PPNetBackbone(MSRBackbone):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims=None,
            scenario_dim=None,
            dropout_rate: float = 0.0,
    ):
        hidden_dims = to_dims(hidden_dims, (input_dim, input_dim))
        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=hidden_dims[-1])
        self.scenario_dim = int(scenario_dim or max(1, input_dim // 3))
        self.agnostic_dim = input_dim - self.scenario_dim
        if self.agnostic_dim <= 0:
            raise ValueError("scenario_dim must be smaller than input_dim")
        self.domain_blocks = nn.ModuleList([
            _PPBlock(
                agnostic_dim=self.agnostic_dim,
                gate_input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_domains)
        ])

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        agnostic_x = x[:, self.scenario_dim:]
        gate_input = torch.cat([x[:, :self.scenario_dim], agnostic_x.detach()], dim=-1)
        outputs = torch.stack([block(agnostic_x, gate_input) for block in self.domain_blocks], dim=1)
        return domain_select(outputs, domain_ids.long())


def create_ppnet(
        input_dim: int = 48,
        num_domains: int = 3,
        batch_size: int = 8,
) -> tuple[PPNetBackbone, torch.Tensor]:
    from betterbole.models.utils.tests import dummy_input_multi_domain

    model = PPNetBackbone(input_dim=input_dim, num_domains=num_domains)
    x, domain_ids = dummy_input_multi_domain(num_domains=num_domains, batch_size=batch_size, emb_size=input_dim)
    out = model(x, domain_ids)
    assert out.shape == (batch_size, model.output_dim)
    return model, out
