import torch

from betterbole.models.msr.backbone.base import MSRBackbone
from betterbole.models.msr.backbone.common import build_mlp, to_dims


class SharedBottomBackbone(MSRBackbone):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims=None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
    ):
        hidden_dims = to_dims(hidden_dims, (input_dim, input_dim))
        super().__init__(input_dim=input_dim, num_domains=num_domains, output_dim=hidden_dims[-1])
        self.bottom = build_mlp(
            input_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        del domain_ids
        return self.bottom(x)


def create_sharedbottom(
        input_dim: int = 48,
        num_domains: int = 3,
        batch_size: int = 8,
) -> tuple[SharedBottomBackbone, torch.Tensor]:
    from betterbole.models.utils.tests import dummy_input_multi_domain

    model = SharedBottomBackbone(input_dim=input_dim, num_domains=num_domains)
    x, domain_ids = dummy_input_multi_domain(num_domains=num_domains, batch_size=batch_size, emb_size=input_dim)
    out = model(x, domain_ids)
    assert out.shape == (batch_size, model.output_dim)
    return model, out
