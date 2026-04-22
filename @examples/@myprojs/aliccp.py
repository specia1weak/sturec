import torch
from torch.utils.data import DataLoader

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb.schema import SparseEmbSetting, QuantileEmbSetting, \
    SparseSetEmbSetting
from betterbole.emb import SchemaManager
import polars as pl
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator, LogDecorator

from torch import nn

from betterbole.models.backbone.ple import PLE
from betterbole.utils import change_root_workdir

change_root_workdir()
Backbone = PLE

class SimplePLE(nn.Module):
    def __init__(self, schema_manager: SchemaManager):
        super(SimplePLE, self).__init__()
        manager = schema_manager
        self.user_encoder = ProfileEncoder(manager.settings, profile_path=manager.work_dir / manager.USER_PROFILE_NAME,
                                      id_source=FeatureSource.USER_ID, feature_source=FeatureSource.USER).to(device)
        self.item_encoder = ProfileEncoder(manager.settings, profile_path=manager.work_dir / manager.ITEM_PROFILE_NAME,
                                      id_source=FeatureSource.ITEM_ID, feature_source=FeatureSource.ITEM).to(device)
        self.inter_emb_layer = InterSideEmb(manager.settings)

        self.manager = manager
        self.input_dim = manager.source2emb_size(FeatureSource.USER_ID, FeatureSource.USER,
                                                 FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                 FeatureSource.INTERACTION)


        print(f"输入总维度{self.input_dim}")
        self.DOMAIN = manager.domain_field
        num_domains = manager.nums_embedding(self.DOMAIN) - 1
        self.ple = Backbone(self.input_dim, num_domains, expert_dims=(256, 128))
        self.LABEL = manager.label_field

    def concat_embed_input_fields(self, interaction):
        user_emb = self.user_encoder.forward(interaction[self.manager.uid_field], split_by=True)
        item_emb = self.item_encoder.forward(interaction[self.manager.iid_field], split_by=True)
        inter_emb = self.inter_emb_layer.forward(interaction, flat2tensor=True)
        return torch.cat([user_emb, item_emb, inter_emb], dim=-1)

    def forward(self, x, domain_ids):
        return self.ple.forward(x, domain_ids)

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        domain_ids = interaction[self.DOMAIN] - 1
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        return final_logits

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

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, 1) # 显存太低没法拉高dim
    item_setting = SparseEmbSetting("item_id", FeatureSource.ITEM_ID, 1) # 显存太低没法拉高dim
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("domain_id", FeatureSource.INTERACTION, 16),

        QuantileEmbSetting("user_hist_category_count", FeatureSource.USER, 10, 16),
        QuantileEmbSetting("user_hist_shop_count", FeatureSource.USER, 10, 16),
        QuantileEmbSetting("user_hist_brand_count", FeatureSource.USER, 10, 16),
        QuantileEmbSetting("user_hist_intention_count", FeatureSource.USER, 10, 16),

        SparseEmbSetting("user_profile_id", FeatureSource.USER, 2),
        SparseEmbSetting("user_profile_group_id", FeatureSource.USER, 2),
        SparseEmbSetting("user_age_id", FeatureSource.USER, 16),
        SparseEmbSetting("user_consumption_level_1", FeatureSource.USER, 16),
        SparseEmbSetting("user_consumption_level_2", FeatureSource.USER, 16),
        SparseEmbSetting("user_is_working", FeatureSource.USER, 16),
        SparseEmbSetting("user_geography_info", FeatureSource.USER, 16),

        SparseEmbSetting("item_category_id", FeatureSource.ITEM, 16),
        SparseEmbSetting("item_shop_id", FeatureSource.ITEM, 16),
        SparseEmbSetting("item_brand_id", FeatureSource.ITEM, 16),

        SparseSetEmbSetting(
            "item_intention_node_id", FeatureSource.ITEM,16,is_string_format=False
        ),
    ]

    manager = SchemaManager(settings_list, "aliccp-workdir", label_fields=["click", "purchase"], domain_fields="domain_id")
    from betterbole.datasets.aliccp import AliCCPDataset

    train_lf = AliCCPDataset.TRAIN_INTER_LF.with_columns(dataset=pl.lit("train"))
    test_lf = AliCCPDataset.TEST_INTER_LF.with_columns(dataset=pl.lit("test"))
    whole_lf = pl.concat([train_lf, test_lf], how="vertical")
    transformed_lf = manager.prepare_data(whole_lf)
    manager.generate_profiles(transformed_lf)
    train_lf = transformed_lf.filter(pl.col("dataset") == "train")
    valid_lf = transformed_lf.filter(pl.col("dataset") == "test")
    train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf)

    # train_lf = AliCCPDataset.TRAIN_INTER_LF.with_columns(dataset=pl.lit("train"))
    # transformed_lf = manager.prepare_data(train_lf)
    # manager.generate_profiles(transformed_lf)
    print("架构编译成功，可供调用。")
    evaluator = LogDecorator(Evaluator("AUC"), save_path=manager.work_dir / "logs.log", title=Backbone.__name__)
    from betterbole.emb.emblayer import ProfileEncoder, InterSideEmb

    ps_dataset = ParquetStreamDataset(train_path, manager.fields()) # 更少的读取
    ps_valid = ParquetStreamDataset(valid_path, manager.fields(), batch_size=4096 * 2, shuffle_and_drop_last=False) # 更少的读取
    ps_dataloader = DataLoader(
        ps_dataset,
        batch_size=None,  # 必须为 None，让 Dataset 自己划分 Batch
        num_workers=0,  # 开启 2 个进程并行读取
        pin_memory=True  # 测试时不占用显存，如果是真实 GPU 训练设为 True
    )

    # puis = PolarsUISampler(
    #         num_items=item_setting.num_embeddings,
    #         user_id_lf=transformed_lf.select(pl.col(manager.uid_field)),
    #         item_id_lf=transformed_lf.select(pl.col(manager.iid_field)),
    #         distribution="uniform"
    #     )

    from betterbole.utils import CudaNamedTimer
    ntr = CudaNamedTimer()
    model = SimplePLE(manager).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

     # 良好的习惯：开启训练模式
    print("开始训练")
    for epoch in range(10):
        total_loss = 0.
        batch_count = 0
        model.train()
        with ntr("epoch"):
            for batch_interaction in ps_dataset:
                with ntr("prepare"):
                    uids = batch_interaction[manager.uid_field]
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
