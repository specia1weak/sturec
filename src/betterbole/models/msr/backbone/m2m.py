import torch
from torch import nn

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import build_mlp, to_dims


class M2MBackbone(MSRBackbone):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_experts: int = 4,
            domain_emb_dim=None,
            nhead: int = 4,
            num_encoder_layers: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        domain_emb_dim = int(domain_emb_dim or expert_dims[-1])
        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=expert_dims[-1])
        self.num_experts = int(num_experts)
        self.domain_embedding = nn.Embedding(num_domains, domain_emb_dim)
        self.input_project = nn.Linear(input_dim, input_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=max(input_dim * 2, expert_dims[-1]),
                dropout=dropout_rate,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
        self.experts = nn.ModuleList([
            build_mlp(
                input_dim,
                expert_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_experts)
        ])
        self.domain_project = build_mlp(
            domain_emb_dim,
            (expert_dims[-1],),
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=False,
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(expert_dims[-1] * 2, self.num_experts),
            nn.Softmax(dim=-1),
        )
        self.meta_tower = nn.Sequential(
            nn.Linear(expert_dims[-1] * 2, expert_dims[-1]),
            nn.ReLU(),
            nn.Linear(expert_dims[-1], expert_dims[-1]),
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        transformed = self.input_project(x).unsqueeze(1)
        transformed = self.transformer(transformed).squeeze(1)
        expert_outs = torch.stack([expert(transformed) for expert in self.experts], dim=1)
        domain_state = self.domain_project(self.domain_embedding(domain_ids.long()))
        pooled_hint = expert_outs.mean(dim=1)
        gate = self.meta_gate(torch.cat([pooled_hint, domain_state], dim=-1)).unsqueeze(-1)
        mixed = torch.sum(gate * expert_outs, dim=1)
        return self.meta_tower(torch.cat([mixed, domain_state], dim=-1)) + mixed


def create_m2m(
        input_dim: int = 48,
        num_domains: int = 3,
        batch_size: int = 8,
) -> tuple[M2MBackbone, torch.Tensor]:
    from betterbole.models.utils.tests import dummy_input_multi_domain

    model = M2MBackbone(input_dim=input_dim, num_domains=num_domains)
    x, domain_ids = dummy_input_multi_domain(num_domains=num_domains, batch_size=batch_size, emb_size=input_dim)
    out = model(x, domain_ids)
    assert out.shape == (batch_size, model.output_dim)
    return model, out
