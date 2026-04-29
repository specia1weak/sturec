import time
from dataclasses import dataclass
from typing import Iterable

import torch

from betterbole.core.train.context import TrainerDataLoaders, TrainerComponents
from betterbole.core.train.trainer import BaseTrainer
from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb.schema import SparseEmbSetting, SparseSetEmbSetting
from betterbole.emb import SchemaManager
import polars as pl
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator

from betterbole.evaluate.manager import EvaluatorManager, DomainFilter
from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.models.base import BaseModel
from betterbole.models.msr import HierRec, build_model
from betterbole.utils.optimize import split_params_by_decay
from betterbole.experiment import change_root_workdir
from betterbole.utils.sequential import extract_history_sequences

change_root_workdir()

@dataclass
class KuairandConfig(ConfigBase):
    dataset_name: str = "kuairand"
    seed: int = 2026
    device: str = "cuda"
    max_epochs: int = 1

    batch_size: int = 4096
    id_emb: int = 32
    side_emb: int = 16
    shuffle_buffer_size: int = 2000000

    model: str = "star"


pm = ParamManager(KuairandConfig)
cfg: KuairandConfig = pm.build()
cfg.experiment_name = "model"
print(cfg)
time.sleep(2)

class KuairandTrainer(BaseTrainer):
    def __init__(self, model: BaseModel, optimizer: torch.optim.Optimizer, manager: SchemaManager,
                 loaders: TrainerDataLoaders, components: TrainerComponents, cfg: ConfigBase):
        super().__init__(model, optimizer, manager, loaders, components, cfg)

if __name__ == '__main__':
    from betterbole.utils.task_chain import auto_queue
    auto_queue()

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, cfg.id_emb, min_freq=10, use_oov=True)
    item_setting = SparseEmbSetting("video_id", FeatureSource.ITEM_ID, cfg.id_emb, min_freq=10, use_oov=True)
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("tab", FeatureSource.INTERACTION, cfg.side_emb, padding_zero=False, use_oov=False),

        SparseEmbSetting("user_active_degree", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_live_streamer", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_video_author", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("friend_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("register_days_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("follow_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("fans_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_lowactive_period", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat0", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat1", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat2", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat3", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat4", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat5", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat6", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat7", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat8", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat9", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat10", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat11", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat12", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat13", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat14", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat15", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat16", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat17", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),

        SparseEmbSetting("author_id", FeatureSource.ITEM, cfg.id_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("music_id", FeatureSource.ITEM, cfg.id_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("video_type", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("music_type", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("visible_status", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),

        SparseSetEmbSetting(
            "tag", FeatureSource.ITEM,cfg.side_emb, max_len=5,
            is_string_format=True, separator=",", min_freq=10, use_oov=True
        ),
    ]
    from betterbole.experiment import preset_workdir
    WORKDIR = preset_workdir(cfg.dataset_name)
    manager = SchemaManager(settings_list, WORKDIR, time_field="time_ms", label_fields="is_click", domain_fields="tab")
    from betterbole.datasets.kuairand import KuaiRandDataset

    user_lf = pl.scan_csv(KuaiRandDataset.USER_FEATURES)
    item_lf = pl.scan_csv(KuaiRandDataset.VIDEO_FEATURES)
    inter_lf = pl.scan_csv([KuaiRandDataset.STD_LOG_FORMER_DATA_P1, KuaiRandDataset.STD_LOG_FORMER_DATA_P2, KuaiRandDataset.RAND_LOG_FORMER_DATA])
    whole_lf: pl.LazyFrame = inter_lf.join(item_lf, on="video_id", how="left").join(user_lf, on="user_id", how="left")

    ## 排除部分场景数据
    top_5_df = whole_lf.group_by("tab").len().sort("len", descending=True).head(5).select("tab").collect()
    top_5_list = top_5_df.get_column("tab").to_list()
    whole_lf = whole_lf.filter(pl.col("tab").is_in(top_5_list))
    print(top_5_list)

    ## 序列处理
    # extract_history_sequences(whole_lf, max_seq_len=20, user_col="user_id", time_col="time_ms", feature_mapping={
    #     "video_id": "video_id_seq",
    #
    # }, seq_len_col="seq_len")

    ## 增加新特征
    parsed_date = pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
    whole_lf = whole_lf.with_columns(
        parsed_date.dt.weekday().alias("day_of_week"),
        parsed_date.dt.weekday().is_in([6, 7]).cast(pl.UInt8).alias("is_weekend"),
        (pl.col("hourmin").cast(pl.Int32) // 100).cast(pl.UInt8).alias("hour")
    )
    whole_lf = manager.make_checkpoint(whole_lf, redo=False, sort_by="time_ms")
    print("join表checkpoint完成")
    ## 处理中
    train_raw = whole_lf.filter(pl.col("date") <= 20220506)
    valid_raw = whole_lf.filter(pl.col("date") == 20220507)  # 仅用 20220507
    test_raw = whole_lf.filter(pl.col("date") == 20220508)  # 20220508 作 test

    manager.fit(train_raw, low_memory=True)
    train_lf = manager.transform(train_raw)
    valid_lf = manager.transform(valid_raw)
    test_lf = manager.transform(test_raw)

    print(train_lf.select("date").head(5).collect())
    print(valid_lf.select("date").head(5).collect())
    train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf, test_lf)
    print("架构编译成功，可供调用。")
    # ======================== 模型在这里 ======================================== #
    num_domains = manager.get_setting(manager.domain_field).vocab_size
    model = build_model(manager, num_domains, cfg.model, embed_dim=16)
    # ======================== 数据处理完成 准备trainer信息 ======================== #
    ps_dataset = ParquetStreamDataset(train_path, manager, batch_size=cfg.batch_size, shuffle=True, shuffle_buffer_size=cfg.shuffle_buffer_size) # 更少的读取
    ps_valid = ParquetStreamDataset(valid_path, manager, batch_size=4096 * 2, shuffle=False) # 不能被shuffle
    # ======================== Trainer 准备 =======================#
    evaluator_manager = EvaluatorManager(log_path=WORKDIR / "logs.log", title=cfg.experiment_name)
    overall_evaluator = Evaluator("auc")
    evaluator_manager.register("overall", overall_evaluator,)

    domain_evluators = [Evaluator("auc") for _ in range(num_domains)]
    for domain, domain_evaluator in enumerate(domain_evluators):
        evaluator_manager.register(
            f"domain{domain}",
            domain_evaluator,
            filter_fn=DomainFilter(manager.domain_field, domain),
        )

    params = split_params_by_decay(model.named_parameters(), weight_decay=1e-5, no_decay_keywords=["embedding"])
    optimizer = torch.optim.Adam(params, lr=1e-3)
    trainer = KuairandTrainer(
        model, optimizer, manager, TrainerDataLoaders(
            train=ps_dataset, valid=ps_valid
        ), TrainerComponents(
            evaluator_manager=evaluator_manager,
        ), cfg
    )

    trainer.run()
