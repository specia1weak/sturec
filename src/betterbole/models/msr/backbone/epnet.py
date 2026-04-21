import torch
from torch import nn

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import build_mlp, to_dims


class EPNetBackbone(MSRBackbone):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims=None,
            scenario_dim=None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        hidden_dims = to_dims(hidden_dims, (input_dim, input_dim))
        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=hidden_dims[-1])
        self.scenario_dim = int(scenario_dim or max(1, input_dim // 3))
        self.agnostic_dim = input_dim - self.scenario_dim
        if self.agnostic_dim <= 0:
            raise ValueError("scenario_dim must be smaller than input_dim")
        self.gate = nn.Sequential(
            nn.Linear(input_dim, self.agnostic_dim),
            nn.Sigmoid(),
        )
        self.agnostic_project = build_mlp(
            self.agnostic_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        del domain_ids
        scenario_x = x[:, :self.scenario_dim]
        agnostic_x = x[:, self.scenario_dim:]
        gate_input = torch.cat([scenario_x, agnostic_x.detach()], dim=-1)
        gated = agnostic_x * self.gate(gate_input)
        return self.agnostic_project(gated)


def create_epnet(
        input_dim: int = 48,
        num_domains: int = 3,
        batch_size: int = 8,
) -> tuple[EPNetBackbone, torch.Tensor]:
    from betterbole.models.utils.tests import dummy_input_multi_domain

    model = EPNetBackbone(input_dim=input_dim, num_domains=num_domains)
    x, domain_ids = dummy_input_multi_domain(num_domains=num_domains, batch_size=batch_size, emb_size=input_dim)
    out = model(x, domain_ids)
    assert out.shape == (batch_size, model.output_dim)
    return model, out
