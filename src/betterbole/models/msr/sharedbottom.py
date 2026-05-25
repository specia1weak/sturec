from typing import Iterable

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.backbone.sharedbottom import SharedBottomBackbone
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.container import MultiScenarioContainer
from betterbole.models.utils.general import ModuleFactory
import torch.nn.functional as F


class SharedBottomModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        self.backbone = SharedBottomBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

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


class RIPLEModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            margin=0.1,
            aux_loss_weight=0.1
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim
        self.margin = margin
        self.aux_loss_weight = aux_loss_weight

        self.backbone = SharedBottomBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

        expert_dim = self.backbone.output_dim
        self.expert_dim = expert_dim

        self.experts = MultiScenarioContainer(
            num_domains,
            ModuleFactory.build_expert(self.input_dim + expert_dim, hidout_dims=[self.input_dim, expert_dim])
        )

        self.sha_proj = nn.Linear(expert_dim, expert_dim, bias=False)
        self.spe_proj = nn.Linear(expert_dim, expert_dim, bias=False)

        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=expert_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

        self._cos_sim = None

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        h_sha = self.backbone(x, domain_ids)
        x_spe = torch.concat([h_sha.detach(), x], dim=-1)
        h_spe = self.experts.forward(x_spe, domain_ids)

        # 计算余弦相似度
        self._cos_sim = F.cosine_similarity(h_sha, h_spe, dim=-1)


        output = self.sha_proj(h_sha) + self.spe_proj(h_spe)
        return self.head.forward(output, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss_s = torch.clamp(torch.abs(self._cos_sim) - self.margin, min=0.0).mean()
        return bce_loss + self.aux_loss_weight * loss_s


if __name__ == '__main__':
    from betterbole.models.utils.tests import DummyCls
    riple = RIPLEModel(DummyCls.MANAGER, DummyCls.NUM_DOMAINS)
    loss = riple.calculate_loss(DummyCls.make_interaction())
    print(loss)

