import torch
from torch.utils.data import DataLoader

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb.schema import SparseEmbSetting
from betterbole.emb import SchemaManager
import polars as pl
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator, LogDecorator

from torch import nn

from betterbole.models.backbone.ple import PLE
from betterbole.models.utils.general import MLP
from betterbole.utils import change_root_workdir
change_root_workdir()
Backbone = MLP

class SimpleBPR(nn.Module):
    def __init__(self, schema_manager: SchemaManager,):
        super(SimpleBPR, self).__init__()
        manager = schema_manager
        self.user_emb_layer = UserSideEmb(manager.settings)
        self.item_emb_layer = ItemSideEmb(manager.settings)
        self.inter_emb_layer = InterSideEmb(manager.settings)
        self.manager = manager
        self.LABEL = manager.label_field
        self.input_dim = manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.USER,
                                                FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                FeatureSource.INTERACTION)
        print(f"输入维度, {self.input_dim}")
        self.backbone = PLE(self.input_dim, num_domains=3)
        self.DOMAIN = manager.domain_field

    def concat_embed_input_fields(self, interaction):
        user_emb = self.user_emb_layer.forward(interaction, flat2tensor=True)
        item_emb = self.item_emb_layer.forward(interaction, flat2tensor=True)
        inter_emb = self.inter_emb_layer.forward(interaction, flat2tensor=True)

        left_emb = [emb for emb in [user_emb, item_emb, inter_emb] if emb is not None]
        whole_emb = torch.cat(left_emb, dim=-1)
        return whole_emb

    def forward(self, x, domain_ids):
        return self.backbone.forward(x, domain_ids)

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        domain_ids = interaction[self.DOMAIN]
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        return torch.sigmoid(final_logits)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        domain_ids = interaction[self.DOMAIN]
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss

if __name__ == '__main__':
    from betterbole.utils import auto_queue
    auto_queue()
    device = "cuda"

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, 32)
    item_setting = SparseEmbSetting("item_id", FeatureSource.ITEM_ID, 32)
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("location", FeatureSource.USER, 16),
        SparseEmbSetting("domain", FeatureSource.USER, 16, padding_zero=False),
    ]

    manager = SchemaManager(settings_list, "douban-workdir", time_field="timestamp", label_field="label", domain_field="domain")
    from betterbole.datasets.douban import DoubanDataset
    user_lf = DoubanDataset.USER_LF
    inter_lf = DoubanDataset.MERGED_INTERS_LF
    whole_lf: pl.LazyFrame = inter_lf.join(user_lf, on="user_id", how="left")
    whole_lf = whole_lf.with_columns(
        (pl.col("rating") >= 4).cast(pl.Int8).alias("label")
    )
    transformed_lf = manager.prepare_data(whole_lf)
    manager.generate_profiles(transformed_lf)
    train_path, valid_path, _ = manager.split_dataset(transformed_lf, group_by="domain")
    print("架构编译成功，可供调用。")
    evaluator_domain0 = LogDecorator(Evaluator("AUC"), save_path=manager.work_dir / "logs0.log", title=Backbone.__name__)
    evaluator_domain1 = LogDecorator(Evaluator("AUC"), save_path=manager.work_dir / "logs1.log", title=Backbone.__name__)
    evaluator_domain2 = LogDecorator(Evaluator("AUC"), save_path=manager.work_dir / "logs2.log", title=Backbone.__name__)
    evaluator_all = LogDecorator(Evaluator("AUC"), save_path=manager.work_dir / "logs.log", title=Backbone.__name__)
    from betterbole.emb.emblayer import InterSideEmb, UserSideEmb, ItemSideEmb

    ps_dataset = ParquetStreamDataset(train_path, manager.fields()) # 更少的读取
    ps_valid = ParquetStreamDataset(valid_path, manager.fields(), batch_size=4096 * 2, shuffle_and_drop_last=False) # 更少的读取
    ps_dataloader = DataLoader(
        ps_dataset,
        batch_size=None,  # 必须为 None，让 Dataset 自己划分 Batch
        num_workers=0,  # 开启 2 个进程并行读取
        pin_memory=True  # 测试时不占用显存，如果是真实 GPU 训练设为 True
    )

    from betterbole.utils import CudaNamedTimer
    ntr = CudaNamedTimer()
    model = SimpleBPR(manager).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

     # 良好的习惯：开启训练模式
    print("开始训练")
    for epoch in range(20):
        total_loss = 0.
        batch_count = 0
        model.train()
        with ntr("epoch"):
            for batch_interaction in ps_dataset:
                # 1. 采样：获取负样本
                # 确保传入的是 numpy，且返回后立刻转为 long 并送到指定的 device
                with ntr("prepare"):
                    uids = batch_interaction[manager.uid_field]
                    # sampled_neg = puis.sample_by_key_ids(uids.numpy(), 1)
                    # batch_interaction[model.neg_sample] = sampled_neg.view(-1)
                    batch_interaction = batch_interaction.to(device)

                # 4. 前向传播与优化
                with ntr("train"):
                    optimizer.zero_grad()
                    loss = model.calculate_loss(batch_interaction)
                    loss.backward()
                    optimizer.step()

                # 5. 累加损失（必须用 .item() 否则会 OOM）
                with ntr("add_loss"):
                    total_loss += loss.item()

                batch_count += 1
                if batch_count % 100 == 0:
                    ntr.report()
                    print(f"Epoch {epoch}, Batch {batch_count}, Current Loss: {loss.item():.4f}")

        model.eval()
        evaluators = [evaluator_domain0, evaluator_domain1, evaluator_domain2]
        with torch.no_grad():
            for batch_interaction in ps_valid:
                # 1. 先整体推到 GPU
                batch_interaction = batch_interaction.to(device)

                # 2. 干净地取出 GPU 上的特征
                uids = batch_interaction[manager.uid_field]
                labels = batch_interaction[manager.label_field]
                scores = model.predict(batch_interaction)
                domain_id = batch_interaction[manager.domain_field]

                # 总domain 指标
                evaluator_all.collect(uids, labels, batch_preds_1d=scores)
                # 分domain 指标
                domain_splits = []
                unique_domains = torch.tensor([0, 1, 2], device=uids.device)
                for d in unique_domains:
                    mask = (domain_id == d)
                    domain_splits.append([uids[mask], labels[mask], scores[mask]])

                for evaluator, (uids, labels, scores) in zip(evaluators, domain_splits):
                    evaluator.collect_pointwise(uids, labels, batch_preds_1d=scores)

        for evaluator in evaluators:
            print(f"Validation Metrics: {evaluator.summary(epoch)}")
            evaluator.clear()
        print(f"Validation Metrics: {evaluator_all.summary(epoch)}")
        ntr.report()
