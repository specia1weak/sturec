import abc
from abc import abstractmethod

from betterbole.core.interaction import Interaction
from betterbole.emb import SchemaManager
from betterbole.experiment.param import ConfigBase
from betterbole.experiment.train.context import TrainContext, TrainerDataLoaders, TrainerComponents
import torch

from betterbole.models.base import BaseModel


class ICustomTrainStep(abc.ABC):
    @abstractmethod
    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        raise NotImplementedError("继承了 ICustomTrainStep 就必须实现此方法！")

class BaseTrainer:
    def __init__(self,
                 model: BaseModel,
                 optimizer: torch.optim.Optimizer,
                 manager: SchemaManager,
                 loaders: TrainerDataLoaders,
                 components: TrainerComponents,
                 cfg: ConfigBase
                 ):
        """
        统一的基座 Trainer。
        components: 包含 Evaluator, Recorder, Timer 等全局组件的字典
        """
        self.model = model
        self.optimizer = optimizer
        self.manager = manager
        self.train_loader = loaders.train
        self.valid_loader = loaders.valid

        # 统一管理的组件
        self.evaluator = components.evaluator
        self.timer = components.timer
        self.cfg = cfg

        # 统计量
        self.epoch = 0
        self.global_step = 0

    def predict_step(self, batch_interaction):
        return self.model.predict(batch_interaction)

    def default_train_step(self, batch, ctx: TrainContext):
        loss = self.model.calculate_loss(batch)
        loss.backward()
        self.optimizer.step()

    def train_epoch(self):
        self.model.train()

        for batch_idx, batch_interaction in enumerate(self.train_loader):
            batch_interaction = batch_interaction.to(self.cfg.device)
            self.optimizer.zero_grad(set_to_none=True)

            ctx = TrainContext(
                epoch=self.epoch,
                global_step=self.global_step,
                batch_idx=batch_idx,
                optimizer=self.optimizer,
                manager=self.manager,
                cfg=self.cfg,
                timer=self.timer
            )

            if isinstance(self.model, ICustomTrainStep):
                self.model.custom_train_step(batch_interaction, ctx)
            else:
                self.default_train_step(batch_interaction, ctx)

            self.global_step += 1
        self.epoch += 1


    @torch.no_grad()
    def evaluate_epoch(self):
        if not self.valid_loader or not self.evaluator:
            return None

        self.model.eval()
        for batch_interaction in self.valid_loader:
            batch_interaction = batch_interaction.to(self.cfg.device)
            uids = batch_interaction[self.manager.uid_field]
            labels = batch_interaction[self.manager.label_field]
            scores = self.predict_step(batch_interaction)
            self.evaluator.collect(uids, labels, batch_preds_1d=scores)

        metrics_result = self.evaluator.summary(self.epoch)
        print(f"Validation Metrics: {metrics_result}")
        self.evaluator.clear()
        return metrics_result

    def run(self):
        for _ in range(self.cfg.epochs):
            self.train_epoch()
            metrics = self.evaluate_epoch()



class TestModel(BaseModel, ICustomTrainStep):
    def __init__(self):
        super().__init__()

    def calculate_loss(self, interaction: Interaction):
        return torch.tensor(0.)

    def predict(self, interaction: Interaction):
        B = len(interaction)
        return torch.ones(B)

    def custom_train_step(self, interaction: Interaction, ctx: TrainContext):
        return


if __name__ == '__main__':
    pass
