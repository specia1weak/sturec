from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.interaction import Interaction
from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import InterSideEmb, ItemSideEmb, UserSideEmb
from betterbole.models.msr.backbone import build_backbone


def build_mlp(
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
) -> nn.Module:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(last_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        last_dim = hidden_dim
    if output_dim is None:
        return nn.Identity() if not layers else nn.Sequential(*layers)
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class DiffMSRIdModel(nn.Module):
    def __init__(
            self,
            schema_manager: SchemaManager,
            domain_head_ids: Tuple[int, ...],
            embedding_size: int = 16,
            shared_bottom_hidden_dims: Tuple[int, ...] = (256, 256),
            domain_head_hidden_dims: Tuple[int, ...] = (256,),
            head_dropout: float = 0.0,
            backbone_name: str = "sharedbottom",
            backbone_params: Optional[dict] = None,
            stage4_freeze_embeddings: bool = True,
            stage4_freeze_shared_bottom: bool = True,
    ):
        super().__init__()
        self.manager = schema_manager
        self.USER = schema_manager.uid_field
        self.ITEM = schema_manager.iid_field
        self.DOMAIN = schema_manager.domain_field
        self.LABEL = schema_manager.label_field
        self.domain_head_ids = tuple(int(domain_id) for domain_id in domain_head_ids)
        self.domain_to_head_idx = {
            domain_id: head_idx for head_idx, domain_id in enumerate(self.domain_head_ids)
        }
        self.backbone_name = backbone_name
        self.backbone_params = dict(backbone_params or {})
        self.stage4_freeze_embeddings = stage4_freeze_embeddings
        self.stage4_freeze_shared_bottom = stage4_freeze_shared_bottom

        self.user_side_emb = UserSideEmb(schema_manager.settings)
        self.item_side_emb = ItemSideEmb(schema_manager.settings)
        self.inter_side_emb = InterSideEmb(schema_manager.settings)

        self.user_emb = self.user_side_emb.embedding.emb_modules[self.USER]
        self.item_emb = self.item_side_emb.embedding.emb_modules[self.ITEM]
        self.domain_emb = self.inter_side_emb.embedding.emb_modules[self.DOMAIN]

        input_dim = embedding_size * 3
        resolved_backbone_params = dict(self.backbone_params)
        if self.backbone_name.lower() in {"sharedbottom", "shared_bottom"}:
            resolved_backbone_params.setdefault("hidden_dims", shared_bottom_hidden_dims)
            resolved_backbone_params.setdefault("dropout_rate", head_dropout)
            resolved_backbone_params.setdefault("batch_norm", True)
        self.backbone = build_backbone(
            self.backbone_name,
            input_dim=input_dim,
            num_domains=len(self.domain_head_ids),
            **resolved_backbone_params,
        )
        self.shared_bottom = self.backbone
        shared_output_dim = self.backbone.output_dim
        self.domain_heads = nn.ModuleList([
            build_mlp(shared_output_dim, domain_head_hidden_dims, output_dim=1, dropout=head_dropout)
            for _ in self.domain_head_ids
        ])

    def embed_user_item_pair(self, interaction: Interaction) -> torch.Tensor:
        user_e = self.user_emb(interaction[self.USER].long())
        item_e = self.item_emb(interaction[self.ITEM].long())
        return torch.stack([user_e, item_e], dim=1)

    def split_pair_embedding(self, pair_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return pair_embedding[:, 0, :], pair_embedding[:, 1, :]

    def embed_triplet(
            self,
            interaction: Interaction,
            domain_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        domain_ids = interaction[self.DOMAIN].long()
        if domain_override is not None:
            domain_ids = torch.full_like(domain_ids, domain_override)
        domain_e = self.domain_emb(domain_ids)
        user_e = self.user_emb(interaction[self.USER].long())
        item_e = self.item_emb(interaction[self.ITEM].long())
        return domain_e, user_e, item_e

    def map_domain_ids_to_head_indices(self, domain_ids: torch.Tensor) -> torch.Tensor:
        head_indices = torch.full_like(domain_ids, -1)
        for domain_id, head_idx in self.domain_to_head_idx.items():
            head_indices = torch.where(
                domain_ids == domain_id,
                torch.full_like(head_indices, head_idx),
                head_indices,
            )
        if (head_indices < 0).any().item():
            unknown = torch.unique(domain_ids[head_indices < 0]).tolist()
            raise ValueError(f"存在未配置 head 的 domain ids: {unknown}, configured={self.domain_head_ids}")
        return head_indices

    def shared_representation_from_embeddings(
            self,
            domain_e: torch.Tensor,
            user_e: torch.Tensor,
            item_e: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([domain_e, user_e, item_e], dim=-1)
        return self.backbone(features, domain_ids.long())

    def logits_from_embeddings(
            self,
            domain_e: torch.Tensor,
            user_e: torch.Tensor,
            item_e: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        head_indices = self.map_domain_ids_to_head_indices(domain_ids.long())
        shared_rep = self.shared_representation_from_embeddings(domain_e, user_e, item_e, head_indices)
        all_logits = torch.cat([head(shared_rep) for head in self.domain_heads], dim=1)
        return all_logits.gather(1, head_indices.unsqueeze(-1)).squeeze(-1)

    def logits(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        domain_ids = interaction[self.DOMAIN].long()
        if domain_override is not None:
            domain_ids = torch.full_like(domain_ids, domain_override)
        return self.logits_from_embeddings(
            *self.embed_triplet(interaction, domain_override=domain_override),
            domain_ids=domain_ids,
        )

    def calculate_loss(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        logits = self.logits(interaction, domain_override=domain_override)
        return F.binary_cross_entropy_with_logits(logits, interaction[self.LABEL].float())

    @torch.no_grad()
    def predict(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        return torch.sigmoid(self.logits(interaction, domain_override=domain_override))

    def configure_stage4_trainability(
            self,
            freeze_embeddings: Optional[bool] = None,
            freeze_shared_bottom: Optional[bool] = None,
    ) -> None:
        freeze_embeddings = self.stage4_freeze_embeddings if freeze_embeddings is None else freeze_embeddings
        freeze_shared_bottom = self.stage4_freeze_shared_bottom if freeze_shared_bottom is None else freeze_shared_bottom
        for param in self.parameters():
            param.requires_grad = True
        if freeze_embeddings:
            for module in (self.user_side_emb, self.item_side_emb, self.inter_side_emb):
                for param in module.parameters():
                    param.requires_grad = False
        if freeze_shared_bottom:
            for param in self.backbone.parameters():
                param.requires_grad = False
        for head in self.domain_heads:
            for param in head.parameters():
                param.requires_grad = True


class DomainClassifier(nn.Module):
    def __init__(self, field_dim: int, num_fields: int = 2):
        super().__init__()
        self.input_dim = field_dim * num_fields
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, pair_embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(pair_embedding.view(pair_embedding.size(0), -1)).squeeze(-1))
