import numpy as np
import torch
from torch.nn import BatchNorm1d
from torch.utils.data import DataLoader

from src.betterbole.data.dataset import ParquetStreamDataset
from src.betterbole.emb.schema import SchemaManager, SparseEmbSetting, QuantileEmbSetting, \
    SparseSetEmbSetting, IdSeqEmbSetting, MinMaxDenseSetting
import polars as pl
from src.betterbole.enum_type import FeatureSource
from src.betterbole.evaluate.evaluator import Evaluator, LogDecorator
from src.betterbole.interaction import Interaction
from src.betterbole.plutils import extract_history_items, extract_history_dict
from src.betterbole.sample import PolarsUISampler

from torch import nn
import torch.nn.functional as F


from src.model.mmoe import SingleLayerMMoE
from src.model.ple import PLE, PLEVersion1, PLEVersion2, PLEVersion3, PLEVersion4
from src.model.star import STAR, StarPle
from src.model.shabtm import SharedBottomLess, SharedBottomPlus
from src.model.utils.general import MLP, ModuleFactory
from src.model.utils.sequence import AttentionSequencePoolingLayer, SequencePoolingLayer
from src.utils import change_root_workdir
from src.utils.monitor import ExplicitFeatureMonitor

change_root_workdir()
Backbone = MLP
EMB_DIM = 16

class SimpleBPR(nn.Module):
    def __init__(self, schema_manager: SchemaManager,):
        super(SimpleBPR, self).__init__()
        manager = schema_manager
        self.user_profile_encoder = UserProfileEncoder(manager.settings, manager.work_dir / manager.USER_PROFILE_NAME)
        self.item_profile_encoder = ItemProfileEncoder(manager.settings, manager.work_dir / manager.ITEM_PROFILE_NAME)
        self.inter_emb_layer = InterSideEmb(manager.settings)

        self.manager = manager
        item_embedding_size = manager.source2emb_size(
            FeatureSource.ITEM_ID, FeatureSource.ITEM,
        )
        self.seq_embeder = SeqEmbedder("history", manager.settings, self.item_profile_encoder)
        self.seq_encoder = AttentionSequencePoolingLayer(item_embedding_size)
        # self.seq_encoder = SequencePoolingLayer(mode='mean')
        self.LABEL = manager.label_field
        self.input_dim = manager.source2emb_size(FeatureSource.USER_ID, FeatureSource.USER,
                                                 FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                 FeatureSource.INTERACTION) + item_embedding_size # 后者来自序列建模
        # self.input_dim = manager.source2emb_size(FeatureSource.USER_ID, FeatureSource.USER,
        #                                          FeatureSource.ITEM_ID, FeatureSource.ITEM,
        #                                          FeatureSource.INTERACTION) # 后者来自序列建模
        self.seq_ln = nn.LayerNorm(item_embedding_size)
        print(f"输入维度, {self.input_dim}")
        self.mlp = MLP(self.input_dim, self.input_dim // 2, 64, 1)
        self.DOMAIN = "domain_id"

    def concat_embed_input_fields(self, interaction):
        uid = interaction[self.manager.uid_field]
        iid = interaction[self.manager.iid_field]
        user_emb = self.user_profile_encoder.forward(uid, flat2tensor=True)
        item_emb = self.item_profile_encoder.forward(iid, flat2tensor=True)
        inter_emb = self.inter_emb_layer.forward(interaction, flat2tensor=True)

        item_emb_seq, seq_len = self.seq_embeder.forward(interaction, flat2tensor=True)
        # seq_emb = self.seq_encoder.forward(item_emb_seq, seq_len)
        seq_emb = self.seq_encoder.forward(item_emb, item_emb_seq, seq_len)
        seq_emb = self.seq_ln(seq_emb)
        # seq_emb = None

        left_emb = [emb for emb in [user_emb, item_emb, inter_emb, seq_emb] if emb is not None]
        whole_emb = torch.cat(left_emb, dim=-1)
        return whole_emb

    def forward(self, x, domain_ids):
        return self.mlp.forward(x).squeeze(-1)

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

class SpecialModel(nn.Module):
    def __init__(self, schema_manager: SchemaManager,):
        super(SpecialModel, self).__init__()
        manager = schema_manager
        self.user_profile_encoder = UserProfileEncoder(manager.settings, manager.work_dir / manager.USER_PROFILE_NAME)
        self.item_profile_encoder = ItemProfileEncoder(manager.settings, manager.work_dir / manager.ITEM_PROFILE_NAME)
        self.inter_emb_layer = InterSideEmb(manager.settings)

        self.manager = manager
        self.LABEL = manager.label_field
        self.DOMAIN = manager.domain_field

        self.whole_input_dim = manager.source2emb_size(FeatureSource.USER_ID, FeatureSource.USER,
                                                 FeatureSource.ITEM_ID, FeatureSource.ITEM,FeatureSource.INTERACTION)
        self.share_input_dim = manager.source2emb_size(FeatureSource.USER,FeatureSource.ITEM,FeatureSource.INTERACTION)
        self.specific_input_dim = manager.source2emb_size(FeatureSource.USER_ID,FeatureSource.ITEM_ID)
        self.item_id_input_dim = manager.source2emb_size(FeatureSource.ITEM_ID)
        self.user_id_input_dim = manager.source2emb_size(FeatureSource.USER_ID)

        # self.whole_expert = MLP(self.whole_input_dim, self.whole_input_dim // 2, 64, 1)
        self.share_expert = MLP(self.share_input_dim, self.share_input_dim // 2, 64, 1)
        self.specific_expert = MLP(self.specific_input_dim, 64, 1)
        self.user_id_expert = MLP(self.user_id_input_dim, 64, 1)
        self.item_id_expert = MLP(self.item_id_input_dim, 64, 1)

        self.cross_expert = MLP(self.item_id_input_dim, 64, 1)
        self.add_expert = MLP(self.item_id_input_dim, 64, 1)

        self.gate_expert = MLP(self.specific_input_dim, 64, 6)

        self.gate_monitor = ExplicitFeatureMonitor()

    def _drop_none(self, list):
        return [emb for emb in list if emb is not None]

    def concat_embed_input_fields(self, interaction):
        uid = interaction[self.manager.uid_field]
        iid = interaction[self.manager.iid_field]
        user_emb_dict = self.user_profile_encoder.forward(uid)
        user_id_emb = user_emb_dict[FeatureSource.USER_ID]
        user_side_emb = user_emb_dict[FeatureSource.USER]

        item_emb_dict = self.item_profile_encoder.forward(iid)
        item_id_emb = item_emb_dict[FeatureSource.ITEM_ID]
        item_side_emb = item_emb_dict[FeatureSource.ITEM]
        inter_emb = self.inter_emb_layer.forward(interaction, flat2tensor=True)

        whole_emb = torch.cat(self._drop_none([user_id_emb, user_side_emb, item_id_emb, item_side_emb, inter_emb]), dim=-1)
        share_emb = torch.cat(self._drop_none([user_side_emb, item_side_emb, inter_emb]) , dim=-1)
        specific_emb = torch.cat(self._drop_none([user_id_emb, item_id_emb]), dim=-1)

        return whole_emb, share_emb, specific_emb, user_id_emb, item_id_emb

    def forward(self, who, sha, spe, uid, iid):
        uid_logits = self.user_id_expert(uid)
        iid_logits = self.item_id_expert(iid)
        sha_logits = self.share_expert(sha)
        spe_logits = self.specific_expert(spe)
        cro_logits = self.cross_expert(uid * iid)
        add_logits = self.add_expert(uid + iid)
        gates = self.gate_expert(spe)

        all_logits = torch.cat([uid_logits, iid_logits, sha_logits, spe_logits, cro_logits, add_logits], dim=-1)

        # 选择mask掉谁
        mask = torch.zeros_like(gates, dtype=torch.bool)
        mask[:, -2:] = True
        mask[:, 2] = True
        gates = gates.masked_fill(mask, float('-inf'))
        # 结束

        gates = F.softmax(gates, dim=-1)
        self.gate_monitor.record("three_expert", gates)
        self.gate_monitor.record("ctr_contribute", gates * all_logits)
        final_logits = torch.sum(gates * all_logits, dim=-1)  # B
        return final_logits

    def predict(self, interaction):
        who, sha, spe, uid, iid = self.concat_embed_input_fields(interaction)
        return self.forward(who, sha, spe, uid, iid)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        final_logits = self.predict(interaction)
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss



if __name__ == '__main__':
    from src.utils.task_chain import auto_queue
    auto_queue()
    device = "cuda"

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, EMB_DIM) # 显存太低没法拉高dim
    item_setting = SparseEmbSetting("movie_id", FeatureSource.ITEM_ID, EMB_DIM) # 显存太低没法拉高dim
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("age", FeatureSource.USER, 16),
        SparseEmbSetting("gender", FeatureSource.USER, 16),
        SparseEmbSetting("occupation", FeatureSource.USER, 16),

        SparseSetEmbSetting("genres", FeatureSource.ITEM, 16),
        IdSeqEmbSetting("history", "history_len", target_setting=item_setting, max_len=50)
    ]

    manager = SchemaManager(settings_list, "movielens-workdir", time_field="timestamp", label_field="label", domain_field="domain_id")
    from src.dataset.movielens import MovieLensDataset
    import pandas as pd
    user_lf = pl.from_pandas(MovieLensDataset.USER_FEATURES_DF).lazy()
    item_lf = pl.from_pandas(MovieLensDataset.ITEM_FEATURES_DF).lazy()
    inter_lf = pl.from_pandas(MovieLensDataset.INTERACTION_DF).lazy()
    whole_lf: pl.LazyFrame = inter_lf.join(item_lf, on="movie_id", how="left").join(user_lf, on="user_id", how="left")
    whole_lf = whole_lf.with_columns(
        pl.col("genres").str.split("|"),
        (pl.col("rating") >= 4).cast(pl.Int8).alias("label")
    )
    whole_lf = whole_lf.with_columns(
        pl.when(pl.col("age") < 25).then(0)
        .when(pl.col("age") < 35).then(1)
        .otherwise(2)
        .alias("domain_id")
    )

    max_seq_len = 50
    whole_lf = extract_history_items(whole_lf, max_seq_len=50,
                                     user_col="user_id",
                                     time_col="timestamp",
                                     item_col="movie_id",
                                     seq_col="history",
                                     seq_len_col="history_len")

    transformed_lf = manager.prepare_data(whole_lf)
    manager.generate_profiles(transformed_lf)
    train_path, valid_path, _ = manager.split_dataset(transformed_lf, strategy="random_ratio")
    # train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf)
    print("架构编译成功，可供调用。")
    evaluator = LogDecorator(Evaluator("AUC"), save_path=manager.work_dir / "logs4backbone.log", title=f"DIM={EMB_DIM},{Backbone.__name__}")
    from src.betterbole.emb.emblayer import ProfileEncoder, InterSideEmb, UserSideEmb, ItemSideEmb, SeqEmbedder, \
        ItemProfileEncoder, UserProfileEncoder

    ps_dataset = ParquetStreamDataset(train_path, manager.fields()) # 更少的读取
    ps_valid = ParquetStreamDataset(valid_path, manager.fields(), batch_size=4096 * 2, shuffle_and_drop_last=False) # 更少的读取
    ps_dataloader = DataLoader(
        ps_dataset,
        batch_size=None,  # 必须为 None，让 Dataset 自己划分 Batch
        num_workers=0,  # 开启 2 个进程并行读取
        pin_memory=True  # 测试时不占用显存，如果是真实 GPU 训练设为 True
    )

    # puis = PolarsUISampler(
    #     num_items=item_setting.num_embeddings,
    #     user_id_lf=transformed_lf.select(pl.col(manager.uid_field)),
    #     item_id_lf=transformed_lf.select(pl.col(manager.iid_field)),
    #     distribution="uniform"
    # )
    model = SpecialModel(manager).to(device)


    from src.utils.time import CudaNamedTimer
    ntr = CudaNamedTimer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(50):
        total_loss = 0.
        batch_count = 0
        model.train()
        with ntr("epoch"):
            for batch_interaction in ps_dataset:
                # 1. 采样：获取负样本
                # 确保传入的是 numpy，且返回后立刻转为 long 并送到指定的 device
                with ntr("prepare"):
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
                    print(model.gate_monitor.get_window_stats())
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