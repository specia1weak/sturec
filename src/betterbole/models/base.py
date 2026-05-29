# from recbole.model.abstract_recommender !!
from abc import abstractmethod, ABC
from dataclasses import dataclass

from betterbole.core.interaction import Interaction
import numpy as np
from torch import nn
import torch
from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import OmniEmbLayer


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor = None # 统一的辅助 Loss 接口

class BaseModel(nn.Module):
    def __init__(self, manager: SchemaManager):
        super(BaseModel, self).__init__()
        self.manager: SchemaManager = manager
        self.omni_embedding: OmniEmbLayer = OmniEmbLayer(manager=manager)

    def apply_xavier_initialization(self) -> None:
        """Optionally reset linear layers to Xavier normal with zero bias."""
        def _reset(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(_reset)

    def calculate_loss(self, interaction: Interaction):
        raise NotImplementedError

    def predict(self, interaction: Interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction: Interaction):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + f": {params}"
        )

