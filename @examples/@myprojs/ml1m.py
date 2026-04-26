import torch
from torch.utils.data import DataLoader

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb.schema import SparseEmbSetting, SparseSetEmbSetting
from betterbole.emb import SchemaManager
import polars as pl
from betterbole.utils import plot_power2_sparsity
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator, LogDecorator

from torch import nn
import torch.nn.functional as F

from betterbole.models.utils.general import MLP, ModuleFactory
from betterbole.models.utils.sequence import AttentionSequencePoolingLayer
from betterbole.utils import change_root_workdir
from betterbole.utils import ExplicitFeatureMonitor
from betterbole.utils import create_optimizer_groups

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
        item_embedding_size = manager.source2emb_dim(
            FeatureSource.ITEM_ID, FeatureSource.ITEM,
        )
        self.seq_embeder = SeqEmbedder("history", manager.settings, self.item_profile_encoder)
        self.seq_encoder = AttentionSequencePoolingLayer(item_embedding_size)
        # self.seq_encoder = SequencePoolingLayer(mode='mean')
        self.LABEL = manager.label_field
        self.input_dim = manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.USER,
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
        user_emb = self.user_profile_encoder.forward(uid)
        item_emb = self.item_profile_encoder.forward(iid)
        inter_emb = self.inter_emb_layer.forward(interaction)

        item_emb_seq, seq_len = self.seq_embeder.forward(interaction)
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

class SimpleMLP(nn.Module):
    def __init__(self, schema_manager: SchemaManager,):
        super(SimpleMLP, self).__init__()
        manager = schema_manager
        self.manager = schema_manager
        self.user_profile_encoder = UserProfileEncoder(manager.settings, manager.work_dir / manager.USER_PROFILE_NAME)
        self.item_profile_encoder = ItemProfileEncoder(manager.settings, manager.work_dir / manager.ITEM_PROFILE_NAME)
        self.LABEL = manager.label_field
        self.DOMAIN = manager.domain_field
        self.input_dim = manager.source2emb_dim(FeatureSource.USER, FeatureSource.USER_ID, FeatureSource.ITEM, FeatureSource.ITEM_ID)

        self.mlp = ModuleFactory.build_expert(self.input_dim, hidout_dims=[256, 128, 64], dropout_rate=0.5)()
        self.tower = nn.Linear(64, 1)

    def forward(self, x):
        interest = self.mlp(x)
        logits = self.tower(interest).squeeze(-1)
        return F.sigmoid(logits)

    def concat_embed_input_fields(self, interaction):
        uid = interaction[self.manager.uid_field]
        iid = interaction[self.manager.iid_field]
        uemb = self.user_profile_encoder.forward(uid)
        iemb = self.item_profile_encoder.forward(iid)
        return torch.cat([uemb, iemb], dim=-1)

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        return self.forward(x)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        final_logits = self.predict(interaction)
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss

class SpecialModel(nn.Module):
    def __init__(self, schema_manager: SchemaManager,):
        super(SpecialModel, self).__init__()
        manager = schema_manager
        self.user_profile_encoder = UserSideEmb(manager.settings)
        self.item_profile_encoder = ItemSideEmb(manager.settings)
        self.inter_emb_layer = InterSideEmb(manager.settings)

        self.manager = manager
        self.LABEL = manager.label_field
        self.DOMAIN = manager.domain_field

        self.whole_input_dim = manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.USER,
                                                      FeatureSource.ITEM_ID, FeatureSource.ITEM, FeatureSource.INTERACTION)
        self.share_input_dim = manager.source2emb_dim(FeatureSource.USER, FeatureSource.ITEM, FeatureSource.INTERACTION)
        self.specific_input_dim = manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.ITEM_ID)
        self.item_id_input_dim = manager.source2emb_dim(FeatureSource.ITEM_ID)
        self.user_id_input_dim = manager.source2emb_dim(FeatureSource.USER_ID)

        # self.whole_expert = MLP(self.whole_input_dim, self.whole_input_dim // 2, 64, 1)
        self.share_expert = MLP(self.share_input_dim, self.share_input_dim // 2, 64, 1)
        self.specific_expert = MLP(self.specific_input_dim, 64, 1)
        self.user_id_expert = MLP(self.user_id_input_dim, 64, 1)
        self.item_id_expert = MLP(self.item_id_input_dim, 64, 1)

        self.cross_expert = MLP(self.item_id_input_dim, 64, 1)
        self.add_expert = MLP(self.item_id_input_dim, 64, 1)

        # self.batch_norm = BatchNorm1d(6)
        self.gate_expert = MLP(self.specific_input_dim, 64, 6)

        self.gate_monitor = ExplicitFeatureMonitor()
        self._train_gate = True
        self._last_gates = None

    def train_gate(self):
        self._train_gate = True
    def freeze_gate(self):
        self._train_gate = False

    def _drop_none(self, list):
        return [emb for emb in list if emb is not None]

    def concat_embed_input_fields(self, interaction):
        user_emb_dict = self.user_profile_encoder.forward(interaction, split_by="source")
        user_id_emb = user_emb_dict[FeatureSource.USER_ID]
        user_side_emb = user_emb_dict[FeatureSource.USER]

        item_emb_dict = self.item_profile_encoder.forward(interaction, split_by="source")
        item_id_emb = item_emb_dict[FeatureSource.ITEM_ID]
        item_side_emb = item_emb_dict[FeatureSource.ITEM]
        inter_emb = self.inter_emb_layer.forward(interaction)

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
        if not self._train_gate:
            gates = gates.detach()

        all_logits = torch.cat([uid_logits, iid_logits, sha_logits, spe_logits, cro_logits, add_logits], dim=-1)

        # 选择mask掉谁
        mask = torch.zeros_like(gates, dtype=torch.bool)
        mask[:, -4:] = True
        mask[:, -2] = False
        gates = gates.masked_fill(mask, float('-inf'))
        # 结束

        gates = F.softmax(gates, dim=-1)
        self.gate_monitor.record("three_expert", gates)
        self.gate_monitor.record("ctr_contribute", gates * all_logits)
        final_logits = torch.sum(gates * all_logits, dim=-1)  # B
        self._last_gates = gates
        return final_logits

    def predict(self, interaction):
        who, sha, spe, uid, iid = self.concat_embed_input_fields(interaction)
        return self.forward(who, sha, spe, uid, iid)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        final_logits = self.predict(interaction)
        gate_loss = -0.02 * torch.var(self._last_gates, dim=0).mean()
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels) + gate_loss
        return loss



if __name__ == '__main__':
    from betterbole.utils import auto_queue
    auto_queue()
    device = "cuda"

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, EMB_DIM, min_freq=10,use_oov=True) # 显存太低没法拉高dim
    item_setting = SparseEmbSetting("movie_id", FeatureSource.ITEM_ID, EMB_DIM, min_freq=10, use_oov=True) # 显存太低没法拉高dim
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("age", FeatureSource.USER, 8),
        SparseEmbSetting("gender", FeatureSource.USER, 8),
        SparseEmbSetting("occupation", FeatureSource.USER, 8),

        SparseSetEmbSetting("genres", FeatureSource.ITEM, 8, min_freq=10, use_oov=True),
        # IdSeqEmbSetting("history", "history_len", target_setting=item_setting, max_len=50)
    ]

    manager = SchemaManager(settings_list, "movielens-workdir", time_field="timestamp", label_fields="label", domain_fields="gender")
    from betterbole.datasets.movielens import MovieLensDataset

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
    plot_power2_sparsity(whole_lf, "user_id", "movie_id", manager.work_dir / "sparsity_power2.png")
    train_raw, valid_raw, test_raw = manager.split_dataset(whole_lf, strategy="sequential_ratio")
    manager.fit(train_raw)
    train_lf = manager.transform(train_raw)
    valid_lf = manager.transform(valid_raw)
    test_lf = manager.transform(test_raw)

    train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf, test_lf)
    print("架构编译成功，可供调用。")
    evaluator = LogDecorator(Evaluator("AUC"),
                             save_path=manager.work_dir / "logs4backbone.log", title=f"DIM={EMB_DIM},{Backbone.__name__}")
    from betterbole.emb.emblayer import InterSideEmb, UserSideEmb, ItemSideEmb, SeqEmbedder, \
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


    from betterbole.utils import CudaNamedTimer
    ntr = CudaNamedTimer()
    named_parameters = create_optimizer_groups(model, weight_decay=1e-5, no_decay_keywords=[]) # 对Embedding施加wd反而更好
    optimizer = torch.optim.Adam(named_parameters, lr=1e-3)
    for epoch in range(50):
        total_loss = 0.
        batch_count = 0
        model.train()
        # if epoch == 25:
        # #     model.train_gate()
        #     print("启动门控")
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
                    if hasattr(model, "gate_monitor"):
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