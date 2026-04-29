import time
from dataclasses import dataclass
from typing import Iterable

import torch

from betterbole.core.train.context import TrainerDataLoaders, TrainerComponents
from betterbole.core.train.trainer import BaseTrainer
from betterbole.data.dataset import ParquetStreamDataset
from betterbole.datasets.taac2026 import TAAC2026Dataset
from betterbole.emb.schema import SparseEmbSetting, SparseSetEmbSetting
from betterbole.emb import SchemaManager
import polars as pl
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator

from torch import nn

from betterbole.evaluate.manager import EvaluatorManager
from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.models.base import BaseModel
from betterbole.models.msr import MSRModel
from betterbole.models.utils.general import ModuleFactory
from betterbole.utils.optimize import split_params_by_decay
from betterbole.experiment import change_root_workdir

change_root_workdir()

@dataclass
class TAACConfig(ConfigBase):
    dataset_name: str = "taac2026"
    seed: int = 2026
    device: str = "cuda"
    max_epochs: int = 100

    batch_size: int = 4096
    id_emb: int = 32
    shuffle_buffer_size: int = 2000000

pm = ParamManager(TAACConfig)
cfg: TAACConfig = pm.build()
# cfg.experiment_name = f"{cfg.backbone}"
print(cfg)
time.sleep(2)

class TAACTrainer(BaseTrainer):
    def __init__(self, model: BaseModel, optimizer: torch.optim.Optimizer, manager: SchemaManager,
                 loaders: TrainerDataLoaders, components: TrainerComponents, cfg: ConfigBase):
        super().__init__(model, optimizer, manager, loaders, components, cfg)

if __name__ == '__main__':
    from betterbole.utils.task_chain import auto_queue
    auto_queue()

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, cfg.id_emb, min_freq=10, use_oov=True)
    item_setting = SparseEmbSetting("item_id", FeatureSource.ITEM_ID, cfg.id_emb, min_freq=10, use_oov=True)

    user_sparse_settings = [SparseEmbSetting(col, FeatureSource.USER, embedding_dim=16, min_freq=10) for col in TAAC2026Dataset.user_sparse_cols]
    item_sparse_settings = [SparseEmbSetting(col, FeatureSource.USER, embedding_dim=16, min_freq=10) for col in TAAC2026Dataset.item_sparse_cols]

    settings_list = [
        user_setting,
        item_setting,
        *user_sparse_settings,
        *item_sparse_settings
    ]

    from betterbole.experiment import preset_workdir
    WORKDIR = preset_workdir(cfg.dataset_name)
    manager = SchemaManager(settings_list, WORKDIR, time_field="timestamp", label_fields="label_type")
    whole_lf: pl.lazyframe = TAAC2026Dataset.WHOLE_LF

    ## 处理中
    train_raw, valid_raw, test_raw = manager.split_dataset(whole_lf, strategy="sequential_ratio")

    manager.fit(train_raw)
    train_lf = manager.transform(train_raw)
    valid_lf = manager.transform(valid_raw)
    test_lf = manager.transform(test_raw)

    print(train_lf.select("timestamp").head(5).collect())
    print(valid_lf.select("timestamp").head(5).collect())
    train_lf = train_lf.sort(by="timestamp")
    train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf, test_lf)
    print("架构编译成功，可供调用。")
    # ======================== 模型在这里 ======================================== #
    class EasyModel(BaseModel):
        def __init__(self, manager: SchemaManager):
            super().__init__(manager)
            self.input_dim = self.omni_embedding.whole.embedding_dim
            self.mlp = ModuleFactory.build_expert(self.input_dim, hidout_dims=(256, 128))()
            self.head = ModuleFactory.build_tower(128)()
            self.LABEL = self.manager.label_field

        def forward(self, x):
            return self.head.forward(self.mlp.forward(x)).squeeze(-1)

        def predict(self, interaction):
            x = self.omni_embedding.whole(interaction)
            final_logits = self.forward(x)
            return final_logits

        def calculate_loss(self, interaction):
            labels = interaction[self.LABEL].float()
            x = self.omni_embedding.whole(interaction)
            final_logits = self.forward(x)
            loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
            return loss


    model = EasyModel(manager).to(cfg.device)
    # ======================== 数据处理完成 准备trainer信息 ======================== #
    ps_dataset = ParquetStreamDataset(train_path, manager.fields(), batch_size=cfg.batch_size, shuffle=True, shuffle_buffer_size=cfg.shuffle_buffer_size, drop_last=False) # 更少的读取
    ps_valid = ParquetStreamDataset(valid_path, manager.fields(), batch_size=4096 * 2, shuffle=False, drop_last=False) # 不能被shuffle

    # ======================== Trainer 准备 =======================#
    overall_evaluator = LogDecorator(Evaluator("auc"), save_path=manager.work_dir / "logs.log", title=cfg.experiment_name)
    evaluator_manager = EvaluatorManager()
    evaluator_manager.register("overall_auc", overall_evaluator)

    params = split_params_by_decay(model.named_parameters(), weight_decay=1e-5, no_decay_keywords=["embedding"])
    optimizer = torch.optim.Adam(params, lr=1e-3)
    trainer = TAACTrainer(
        model, optimizer, manager, TrainerDataLoaders(
            train=ps_dataset, valid=ps_valid
        ), TrainerComponents(
            evaluator_manager=evaluator_manager,
        ), cfg
    )

    trainer.run()
