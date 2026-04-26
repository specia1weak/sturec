import time
from dataclasses import dataclass
from typing import Iterable, Any

import torch
import random
import numpy as np

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb.emblayer import UserSideEmb, InterSideEmb, ItemSideEmb
from betterbole.emb.schema import SparseEmbSetting, SparseSetEmbSetting
from betterbole.emb import SchemaManager
import polars as pl
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator, LogDecorator

from torch import nn

from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.models.msr import M3oEBackbone, M3oEVersion2Backbone, MSRBackbone, PLEBackbone, STARBackbone
from betterbole.models.utils.container import MultiScenarioContainer
from betterbole.models.utils.general import ModuleFactory
from betterbole.utils.optimize import split_params_by_decay
from betterbole.models.backbone.ple import PLE
from betterbole.experiment import change_root_workdir

change_root_workdir()

@dataclass
class KuairandConfig(ConfigBase):
    experiment_name: str = "kuairand-simple"
    dataset_name: str = "kuairand"
    seed: int = 2026
    device: str = "cuda"

    batch_size: int = 4096 * 10
    backbone: MSRBackbone = M3oEBackbone
    m3oe_star_dims: Iterable = (512, 256)
    m3oe_expert_dims: Iterable = (64,)
    m3oe_expert_num: int = 4
    m3oe_factor_update_step: int = 20
    id_emb: int = 32
    shuffle_buffer_size: int = 2000000

pm = ParamManager(KuairandConfig)
pm.register(
    "backbone", {
        "m3oe": M3oEBackbone,
        "ple": PLEBackbone,
        "star": STARBackbone
    }
)
cfg: KuairandConfig = pm.build(
    backbone="ple"
)
print(cfg)
time.sleep(2)

def build_m3oe_tower(in_dim: int):
    return lambda: nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.ReLU(),
        nn.Linear(in_dim, 1),
    )


class Model(nn.Module):
    def __init__(self, schema_manager: SchemaManager):
        super(Model, self).__init__()
        manager = schema_manager
        self.user_emb_layer = UserSideEmb(manager.settings)
        self.item_emb_layer = ItemSideEmb(manager.settings)
        self.inter_emb_layer = InterSideEmb(manager.settings)

        self.manager = manager
        self.input_dim = manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.USER,
                                                FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                FeatureSource.INTERACTION)
        print(f"输入总维度{self.input_dim}")
        self.DOMAIN = manager.domain_field
        num_domains = manager.get_setting(self.DOMAIN).vocab_size
        self.backbone = cfg.backbone(self.input_dim, num_domains, expert_dims=(256, 128))
        self.head = MultiScenarioContainer(num_domains, ModuleFactory.build_tower(128))
        self.LABEL = manager.label_field

    def concat_embed_input_fields(self, interaction):
        user_emb = self.user_emb_layer.forward(interaction)
        item_emb = self.item_emb_layer.forward(interaction)
        inter_emb = self.inter_emb_layer.forward(interaction)
        return torch.cat([user_emb, item_emb, inter_emb], dim=-1)

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

#789537121234984   1000 都锁
#7892546954781408  100 都锁
#7892996929591591  1000-不要gate
#7891              1000 都锁
#7897953494286142
# class M3oE(nn.Module):
#     def __init__(self, schema_manager: SchemaManager):
#         super(M3oE, self).__init__()
#         manager = schema_manager
#         self.user_emb_layer = UserSideEmb(manager.settings)
#         self.item_emb_layer = ItemSideEmb(manager.settings)
#         self.inter_emb_layer = InterSideEmb(manager.settings)
#
#         self.manager = manager
#         self.input_dim = manager.source2emb_size(FeatureSource.USER_ID, FeatureSource.USER,
#                                                  FeatureSource.ITEM_ID, FeatureSource.ITEM,
#                                                  FeatureSource.INTERACTION)
#         print(f"输入总维度{self.input_dim}")
#         self.DOMAIN = manager.domain_field
#         num_domains = manager.get_setting(self.DOMAIN).vocab_size
#         self.num_domains = num_domains
#         self.m3oe = M3OE_BACKBONE(
#             self.input_dim,
#             num_domains,
#             star_dims=M3OE_STAR_DIMS,
#             expert_dims=M3OE_EXPERT_DIMS,
#             num_shared_experts=M3OE_EXPERT_NUM,
#         )
#         self.head = MultiScenarioContainer(num_domains, build_m3oe_tower(self.m3oe.output_dim))
#         self.m3oe.get_parameter_groups()
#         self.LABEL = manager.label_field
#
#     @property
#     def param_groups(self):
#         """
#         在顶层直接划分所有参数。
#         这样既包含了 Emb、M3oE 和 Head，又保持了 (name, param) 的格式供后续正则化工具使用。
#         """
#         arch_named_params = []
#         base_named_params = []
#         for name, param in self.named_parameters():
#             if 'balance_factor' in name or 'balance_gate' in name:
#                 arch_named_params.append((name, param))
#             else:
#                 base_named_params.append((name, param))
#         return arch_named_params, base_named_params
#
#     def get_domain_ids(self, interaction):
#         return interaction[self.DOMAIN].long()
#
#     def concat_embed_input_fields(self, interaction):
#         user_emb = self.user_emb_layer.forward(interaction)
#         item_emb = self.item_emb_layer.forward(interaction)
#         inter_emb = self.inter_emb_layer.forward(interaction)
#         return torch.cat([user_emb, item_emb, inter_emb], dim=-1)
#
#     def forward(self, x, domain_ids):
#         x = self.m3oe.forward(x, domain_ids)
#         return self.head.forward(x, domain_ids).squeeze(-1)
#
#     def predict(self, interaction):
#         x = self.concat_embed_input_fields(interaction)
#         domain_ids = self.get_domain_ids(interaction)
#         x = torch.flatten(x, start_dim=1)
#         final_logits = self.forward(x, domain_ids)
#         return torch.sigmoid(final_logits)
#
#     def calculate_loss(self, interaction):
#         labels = interaction[self.LABEL].float()
#         domain_ids = self.get_domain_ids(interaction)
#         x = self.concat_embed_input_fields(interaction)
#         x = torch.flatten(x, start_dim=1)
#         final_logits = self.forward(x, domain_ids)
#         loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
#         return loss
#

# def _format_factor_values(factor_module):
#     if factor_module is None or not hasattr(factor_module, "balance_factor"):
#         return None
#
#     raw_factor = factor_module.balance_factor
#     if isinstance(raw_factor, nn.Embedding):
#         values = torch.sigmoid(raw_factor.weight.detach()).view(-1).cpu().tolist()
#         return [round(v, 4) for v in values]
#
#     values = torch.sigmoid(raw_factor.detach()).view(-1).cpu().tolist()
#     if len(values) == 1:
#         return round(values[0], 4)
#     return [round(v, 4) for v in values]
#
#
# def print_m3oe_factors(model):
#     backbone = getattr(model, "m3oe", None)
#     if backbone is None:
#         return
#
#     alpha_values = _format_factor_values(getattr(backbone, "domain_expert_factor", None))
#     beta_values = _format_factor_values(getattr(backbone, "domain_balance_factor", None))
#     print(f"[M3oE Factors] exp_d={alpha_values} bal_d={beta_values}")


def set_named_params_requires_grad(named_params, requires_grad: bool):
    for _, param in named_params:
        param.requires_grad_(requires_grad)


if __name__ == '__main__':
    from betterbole.utils.task_chain import auto_queue
    auto_queue()

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, cfg.id_emb, min_freq=10, use_oov=True)
    item_setting = SparseEmbSetting("video_id", FeatureSource.ITEM_ID, cfg.id_emb, min_freq=10, use_oov=True)
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("tab", FeatureSource.INTERACTION, 16, padding_zero=False, use_oov=False),

        SparseEmbSetting("user_active_degree", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("is_live_streamer", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("is_video_author", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("friend_user_num_range", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("register_days_range", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("follow_user_num_range", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("fans_user_num_range", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("is_lowactive_period", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat0", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat1", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat2", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat3", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat4", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat5", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat6", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat7", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat8", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat9", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat10", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat11", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat12", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat13", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat14", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat15", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat16", FeatureSource.USER, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat17", FeatureSource.USER, 16, min_freq=10, use_oov=True),

        SparseEmbSetting("author_id", FeatureSource.ITEM, cfg.id_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("music_id", FeatureSource.ITEM, cfg.id_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("video_type", FeatureSource.ITEM, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("music_type", FeatureSource.ITEM, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("visible_status", FeatureSource.ITEM, 16, min_freq=10, use_oov=True),

        SparseSetEmbSetting(
            "tag", FeatureSource.ITEM,16, max_len=5,
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
    top_5_domains = (
        whole_lf.group_by("tab")
        .len()
        .sort("len", descending=True)
        .head(5)
        .select("tab")
    )
    print(top_5_domains.collect())

    whole_lf = whole_lf.join(top_5_domains, on="tab", how="semi")

    ## 增加新特征
    parsed_date = pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
    whole_lf = whole_lf.with_columns(
        parsed_date.dt.weekday().alias("day_of_week"),
        parsed_date.dt.weekday().is_in([6, 7]).cast(pl.UInt8).alias("is_weekend"),
        (pl.col("hourmin").cast(pl.Int32) // 100).cast(pl.UInt8).alias("hour")
    )

    ## 处理中
    train_raw = whole_lf.filter(pl.col("date") <= 20220506)
    valid_raw = whole_lf.filter(pl.col("date") == 20220507)  # 仅用 20220507
    test_raw = whole_lf.filter(pl.col("date") == 20220508)  # 20220508 作 test

    manager.fit(train_raw)
    train_lf = manager.transform(train_raw)
    valid_lf = manager.transform(valid_raw)
    test_lf = manager.transform(test_raw)

    print(train_lf.select("date").head(5).collect())
    print(valid_lf.select("date").head(5).collect())
    train_lf = train_lf.sort(by="time_ms")
    train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf, test_lf)
    print("架构编译成功，可供调用。")


    # ======================== 数据处理完成 ======================== #


    evaluator = LogDecorator(Evaluator("auc"), save_path=manager.work_dir / "logs.log", title=cfg.experiment_name)
    ps_dataset = ParquetStreamDataset(train_path, manager.fields(), batch_size=cfg.batch_size, shuffle=True, shuffle_buffer_size=cfg.shuffle_buffer_size) # 更少的读取
    ps_valid = ParquetStreamDataset(valid_path, manager.fields(), batch_size=4096 * 2, shuffle=False) # 不能被shuffle

    from betterbole.utils.time import CudaNamedTimer
    ntr = CudaNamedTimer()
    model = Model(manager).to(cfg.device)

    params = split_params_by_decay(model.named_parameters(), weight_decay=1e-5, no_decay_keywords=["embedding"])
    optimizer = torch.optim.Adam(params, lr=1e-3)

    global_step = 0
    for epoch in range(1):
        total_loss = 0.
        batch_count = 0
        model.train()
        with ntr("epoch"):
            for batch_interaction in ps_dataset:
                global_step += 1
                uids = batch_interaction[manager.uid_field]
                batch_interaction = batch_interaction.to(cfg.device)
                with ntr("train"):
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.calculate_loss(batch_interaction)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                batch_count += 1

                if batch_count % 200 == 0:
                    ntr.report()
                    print(f"Epoch {epoch}, Batch {batch_count}, Current Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            for batch_interaction in ps_valid:
                batch_interaction = batch_interaction.to(cfg.device)
                uids = batch_interaction[manager.uid_field]
                labels = batch_interaction[manager.label_field]
                scores = model.predict(batch_interaction)
                evaluator.collect_pointwise(uids, labels, batch_preds_1d=scores)

        metrics_result = evaluator.summary(epoch)
        print(f"Validation Metrics: {metrics_result}")
        evaluator.clear()
        ntr.report()