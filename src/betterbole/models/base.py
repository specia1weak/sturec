# from recbole.model.abstract_recommender !!
from abc import abstractmethod, ABC

from betterbole.core.interaction import Interaction
import numpy as np
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import OmniEmbLayer


class BaseModel(nn.Module):
    def __init__(self, manager: SchemaManager):
        super(BaseModel, self).__init__()
        self.manager: SchemaManager = manager
        self.omni_embedding: OmniEmbLayer = OmniEmbLayer(manager=manager)

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

