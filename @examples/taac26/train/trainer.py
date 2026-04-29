from __future__ import annotations

import torch

from betterbole.core.train.context import TrainerComponents, TrainerDataLoaders
from betterbole.core.train.trainer import BaseTrainer
from betterbole.emb import SchemaManager
from betterbole.experiment.param import ConfigBase
from betterbole.models.base import BaseModel


class TAACTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        manager: SchemaManager,
        loaders: TrainerDataLoaders,
        components: TrainerComponents,
        cfg: ConfigBase,
    ):
        super().__init__(model, optimizer, manager, loaders, components, cfg)
