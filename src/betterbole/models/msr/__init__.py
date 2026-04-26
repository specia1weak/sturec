from typing import Dict, Type

from betterbole.core.enum_type import FeatureSource
from betterbole.core.interaction import Interaction
from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import UserSideEmb, ItemSideEmb, InterSideEmb, OmniEmbLayer
from betterbole.experiment.param import ConfigBase
from betterbole.experiment.train.context import TrainContext
from betterbole.experiment.train.trainer import CustomTrainStepProtocol
from betterbole.models.base import BaseModel
from betterbole.models.msr.backbone import (
    MSRBackbone, PLEBackbone, STARBackbone, SharedBottomBackbone, M3oEBackbone, M3oEVersion1Backbone,
    M3oEVersion2Backbone, MMoEBackbone, M2MBackbone, PPNetBackbone, EPNetBackbone
)
from betterbole.models.msr.base import MSRModel
from betterbole.models.utils.container import MultiScenarioContainer
from betterbole.models.utils.general import ModuleFactory

import torch
from torch import nn

class CustomMSRModel(MSRModel):
    def __init__(self, manager: SchemaManager, num_domains: int, backbone: MSRBackbone):
        super().__init__(manager, num_domains)
        self.backbone = backbone
        self.omni_embedding: OmniEmbLayer = self.omni_embedding
        self.input_dim = self.omni_embedding.whole.embedding_dim
        self.DOMAIN = self.manager.domain_field
        self.backbone = backbone
        self.head = MultiScenarioContainer(num_domains, ModuleFactory.build_tower(self.backbone.output_dim))
        self.LABEL = self.manager.label_field

    def concat_embed_input_fields(self, interaction):
        return self.omni_embedding.whole(interaction)
    def forward(self, x, domain_ids):
        return self.head.forward(self.backbone.forward(x, domain_ids), domain_ids).squeeze(-1)
    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        domain_ids = interaction[self.DOMAIN]
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        return final_logits

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        domain_ids = interaction[self.DOMAIN]
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss


class M3oEModel(MSRModel, CustomTrainStepProtocol):
    def __init__(self, schema_manager: SchemaManager, num_domains: int, backbone: MSRBackbone):
        super().__init__(schema_manager, num_domains, backbone)

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        if ctx.global_step % 30 == 0:
            self.optimizer_arch.zero_grad()

        self.optimizer_base.zero_grad()

        loss.backward()
        if ctx.global_step % 30 == 0:
            self.optimizer_arch.step()
        self.optimizer_base.step()

def build_model(schema_manager: SchemaManager,
                num_domains: int,
                backbone_cls: Type[MSRBackbone],
                backbone_kwargs: Dict, ):
    if isinstance(backbone, M3oEBackbone):
        return M3oEModel(schema_manager, num_domains, backbone_cls(**backbone_kwargs))
    else:
        return CustomMSRModel(schema_manager, num_domains, backbone_cls(**backbone_kwargs))