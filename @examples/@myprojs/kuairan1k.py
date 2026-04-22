import torch

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb.schema import SparseEmbSetting, SparseSetEmbSetting
from betterbole.emb import SchemaManager
import polars as pl
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator, LogDecorator

from torch import nn

from betterbole.models.msr import M3oEBackbone
from betterbole.utils.optimize import split_params_by_decay
from betterbole.models.backbone.ple import PLE
from betterbole.experiment import change_root_workdir

change_root_workdir()
Backbone = PLE
ID_EMB = 32

class SimplePLE(nn.Module):
    def __init__(self, schema_manager: SchemaManager):
        super(SimplePLE, self).__init__()
        manager = schema_manager
        self.user_emb_layer = UserSideEmb(manager.settings)
        self.item_emb_layer = ItemSideEmb(manager.settings)
        self.inter_emb_layer = InterSideEmb(manager.settings)

        self.manager = manager
        self.input_dim = manager.source2emb_size(FeatureSource.USER_ID, FeatureSource.USER,
                                                 FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                 FeatureSource.INTERACTION)
        print(f"输入总维度{self.input_dim}")
        self.DOMAIN = manager.domain_field
        num_domains = manager.get_setting(self.DOMAIN).vocab_size
        self.ple = Backbone(self.input_dim, num_domains, expert_dims=(256, 128))
        self.LABEL = manager.label_field

    def concat_embed_input_fields(self, interaction):
        user_emb = self.user_emb_layer.forward(interaction)
        item_emb = self.item_emb_layer.forward(interaction)
        inter_emb = self.inter_emb_layer.forward(interaction)
        return torch.cat([user_emb, item_emb, inter_emb], dim=-1)

    def forward(self, x, domain_ids):
        return self.ple.forward(x, domain_ids)

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        domain_ids = interaction[self.DOMAIN] - 1
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        return torch.sigmoid(final_logits)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        domain_ids = interaction[self.DOMAIN] - 1
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss


class M3oE(nn.Module):
    def __init__(self, schema_manager: SchemaManager):
        super(M3oE, self).__init__()
        manager = schema_manager
        self.user_emb_layer = UserSideEmb(manager.settings)
        self.item_emb_layer = ItemSideEmb(manager.settings)
        self.inter_emb_layer = InterSideEmb(manager.settings)

        self.manager = manager
        self.input_dim = manager.source2emb_size(FeatureSource.USER_ID, FeatureSource.USER,
                                                 FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                 FeatureSource.INTERACTION)
        print(f"输入总维度{self.input_dim}")
        self.DOMAIN = manager.domain_field
        num_domains = manager.get_setting(self.DOMAIN).vocab_size
        self.m3oe = M3oEBackbone(self.input_dim, num_domains)
        self.m3oe.get_parameter_groups()
        self.LABEL = manager.label_field

    @property
    def param_groups(self):
        return self.m3oe.get_parameter_groups()

    def concat_embed_input_fields(self, interaction):
        user_emb = self.user_emb_layer.forward(interaction)
        item_emb = self.item_emb_layer.forward(interaction)
        inter_emb = self.inter_emb_layer.forward(interaction)
        return torch.cat([user_emb, item_emb, inter_emb], dim=-1)

    def forward(self, x, domain_ids):
        return self.ple.forward(x, domain_ids)

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        domain_ids = interaction[self.DOMAIN] - 1
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        return torch.sigmoid(final_logits)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        domain_ids = interaction[self.DOMAIN] - 1
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss

if __name__ == '__main__':
    from betterbole.utils import auto_queue
    auto_queue()
    device = "cuda"

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, ID_EMB, min_freq=10, use_oov=True)
    item_setting = SparseEmbSetting("video_id", FeatureSource.ITEM_ID, ID_EMB, min_freq=10, use_oov=True)
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

        SparseEmbSetting("author_id", FeatureSource.ITEM, ID_EMB, min_freq=10, use_oov=True),
        SparseEmbSetting("music_id", FeatureSource.ITEM, ID_EMB, min_freq=10, use_oov=True),
        SparseEmbSetting("video_type", FeatureSource.ITEM, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("music_type", FeatureSource.ITEM, 16, min_freq=10, use_oov=True),
        SparseEmbSetting("visible_status", FeatureSource.ITEM, 16, min_freq=10, use_oov=True),

        SparseSetEmbSetting(
            "tag", FeatureSource.ITEM,16, max_len=5,
            is_string_format=True, separator=",", min_freq=10, use_oov=True
        ),
    ]
    from betterbole.experiment import preset_workdir
    WORKDIR = preset_workdir("kuairand")
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
    # ========================
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

    evaluator = LogDecorator(Evaluator("auc"), save_path=manager.work_dir / "logs.log", title=Backbone.__name__)
    from betterbole.emb.emblayer import InterSideEmb, UserSideEmb, ItemSideEmb

    ps_dataset = ParquetStreamDataset(train_path, manager.fields()) # 更少的读取
    ps_valid = ParquetStreamDataset(valid_path, manager.fields(), batch_size=4096 * 2, shuffle=False) # 不能被shuffle

    from betterbole.utils import CudaNamedTimer
    ntr = CudaNamedTimer()
    model = M3oE(manager).to(device)
    arch_params, base_params = model.param_groups
    arch_params_list = split_params_by_decay(arch_params, weight_decay=1e-6, no_decay_keywords=["embedding"])
    base_params_list = split_params_by_decay(base_params, weight_decay=1e-6, no_decay_keywords=["embedding"])

    optimizer_arch = torch.optim.Adam(arch_params_list, lr=1e-4)
    optimizer_base = torch.optim.Adam(base_params, lr=1e-3)

     # 良好的习惯：开启训练模式
    print("开始训练")
    global_step = 0
    for epoch in range(20):
        total_loss = 0.
        batch_count = 0
        model.train()
        with ntr("epoch"):
            for batch_interaction in ps_dataset:
                # 1. 采样：获取负样本
                # 确保传入的是 numpy，且返回后立刻转为 long 并送到指定的 device
                global_step += 1

                with ntr("prepare"):
                    uids = batch_interaction[manager.uid_field]
                    # sampled_neg = puis.sample_by_key_ids(uids.numpy(), 1)
                    # batch_interaction[model.neg_sample] = sampled_neg.view(-1)
                    batch_interaction = batch_interaction.to(device)

                if global_step % 1000 == 0:
                    optimizer_arch.zero_grad()
                    loss = model.calculate_loss(batch_interaction)
                    loss.backward()
                    optimizer_arch.step()

                # 4. 前向传播与优化
                with ntr("train"):
                    optimizer_base.zero_grad()
                    loss = model.calculate_loss(batch_interaction)
                    loss.backward()
                    optimizer_base.step()

                # 5. 累加损失（必须用 .item() 否则会 OOM）
                with ntr("add_loss"):
                    total_loss += loss.item()

                batch_count += 1
                if batch_count % 100 == 0:
                    ntr.report()
                    print(f"Epoch {epoch}, Batch {batch_count}, Current Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            for batch_interaction in ps_valid:
                # 1. 先整体推到 GPU
                batch_interaction = batch_interaction.to(device)

                # 2. 干净地取出 GPU 上的特征
                uids = batch_interaction[manager.uid_field]
                labels = batch_interaction[manager.label_field]

                # 3. 预测并打分
                scores = model.predict(batch_interaction)
                evaluator.collect_pointwise(uids, labels, batch_preds_1d=scores)

        metrics_result = evaluator.summary(epoch)
        print(f"Validation Metrics: {metrics_result}")
        evaluator.clear()
        ntr.report()
        # print(f"=== Epoch {epoch} Done, Average Loss: {total_loss / batch_count:.4f} ===")
