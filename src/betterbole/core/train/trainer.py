from pathlib import Path
from typing import Optional, Union

from betterbole.core.interaction import Interaction
from betterbole.emb import SchemaManager
from betterbole.experiment.param import ConfigBase
from betterbole.core.train.context import TrainContext, TrainerDataLoaders, TrainerComponents
from betterbole.core.train.hooks import CustomTrainStepProtocol, TrainerHooksProtocol
import torch

from betterbole.models.base import BaseModel

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
        self.model = model.to(cfg.device)
        self.optimizer = optimizer
        self.manager = manager
        self.train_loader = loaders.train
        self.valid_loader = loaders.valid

        # 统一管理的组件
        self.evaluator = components.evaluator_manager
        self.timer = components.timer
        self.recorder = components.recorder
        self.early_stepper = components.early_stepper
        self.cfg = cfg

        # 统计量
        self.epoch = 0
        self.global_step = 0

    def predict_step(self, batch_interaction):
        return self.model.predict(batch_interaction)

    def save_checkpoint(self, tag: str = "", metrics: Optional[dict] = None) -> Union[Path, None]:
        ckpt_dir = getattr(self.cfg, "ckpt_dir", "")
        if not ckpt_dir:
            return None

        save_dir = Path(ckpt_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        safe_tag = f"_{tag}" if tag else ""
        filename = f"epoch_{self.epoch:04d}_step_{self.global_step:08d}{safe_tag}.pt"
        save_path = save_dir / filename

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "experiment_name": self.cfg.experiment_name,
            "dataset_name": self.cfg.dataset_name,
        }
        torch.save(checkpoint, save_path)
        return save_path

    def default_train_step(self, batch, ctx: TrainContext):
        loss = self.model.calculate_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self):
        self.model.train()
        epoch_ctx = TrainContext(
            epoch=self.epoch,
            global_step=self.global_step,
            batch_idx=-1,
            optimizer=self.optimizer,
            manager=self.manager,
            cfg=self.cfg,
            timer=self.timer,
            recorder=self.recorder
        )
        if isinstance(self.model, TrainerHooksProtocol):
            self.model.on_train_epoch_start(epoch_ctx)

        last_ctx = epoch_ctx
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
                timer=self.timer,
                recorder=self.recorder
            )
            last_ctx = ctx

            if isinstance(self.model, CustomTrainStepProtocol):
                loss = self.model.custom_train_step(batch_interaction, ctx)
            else:
                loss = self.default_train_step(batch_interaction, ctx)

            if self.global_step % 100 == 0:
                print(f"step:{self.global_step} loss:{loss}")

            self.global_step += 1
        if isinstance(self.model, TrainerHooksProtocol):
            self.model.on_train_epoch_end(last_ctx)
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
            self.evaluator.collect(uids, labels, batch_interaction, batch_preds_1d=scores)

        metrics_result = self.evaluator.summary(self.epoch)
        if isinstance(self.model, TrainerHooksProtocol):
            eval_ctx = TrainContext(
                epoch=self.epoch,
                global_step=self.global_step,
                batch_idx=-1,
                optimizer=self.optimizer,
                manager=self.manager,
                cfg=self.cfg,
                timer=self.timer,
                recorder=self.recorder
            )
            self.model.on_eval_epoch_end(metrics_result, eval_ctx)
        print(f"Validation Metrics: {metrics_result}")
        self.evaluator.clear()
        return metrics_result

    def run(self):
        for _ in range(self.cfg.max_epochs):
            self.train_epoch()
            metrics = self.evaluate_epoch()
            print(metrics)
            if self.early_stepper:
                is_best, should_stop = self.early_stepper.step(metrics, epoch=self.epoch)
                if is_best:
                    save_path = self.save_checkpoint(tag="best", metrics=metrics)
                    print(f"[EarlyStop] best at epoch {self.epoch}")
                    if save_path is not None:
                        print(f"[Checkpoint] saved: {save_path}")
                if should_stop:
                    print(f"[EarlyStop] stop at epoch {self.epoch}")
                    break



class TestModel(BaseModel):
    def __init__(self):
        super().__init__()

    def predict(self, interaction: Interaction):
        pass

    def calculate_loss(self, interaction: Interaction):
        pass

    def custom_train_step(self, interaction: Interaction, ctx: TrainContext):
        return


if __name__ == '__main__':
    print(isinstance(TestModel(), CustomTrainStepProtocol))
    TestModel.from_kwargs(i=1)
