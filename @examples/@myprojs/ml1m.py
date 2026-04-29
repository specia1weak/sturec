from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb.schema import SparseEmbSetting, SparseSetEmbSetting, SeqGroupEmbSetting, SharedVocabSeqSetting, \
    SeqGroupConfig
from betterbole.emb import SchemaManager
import polars as pl

from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.models.utils.transfomer import Transformer, TransformerEncoder, \
    PositionalEmbedding, RotaryPositionalEmbedding, LearnablePositionalEmbedding
from betterbole.utils.sequential import extract_history_sequences
from betterbole.utils.visualize import plot_power2_sparsity
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator, LogDecorator

from torch import nn
import torch.nn.functional as F

from betterbole.models.utils.general import MLP, ModuleFactory, BifurcatedMLP, FeatureBifurcator
from betterbole.models.utils.sequence import AttentionSequencePoolingLayer, SequencePoolingLayer
from betterbole.experiment import change_root_workdir, preset_workdir
from betterbole.utils.recorder import ExplicitFeatureRecorder
from betterbole.utils.optimize import split_params_by_decay
from betterbole.models.base import BaseModel

change_root_workdir()

@dataclass
class MovieLensConfig(ConfigBase):
    experiment_name: str = "ml1m"
    dataset_name: str = "movie-lens"
    emb_dim: int = 16

cfg: MovieLensConfig = ParamManager(MovieLensConfig).build()
print(cfg.dataset_name)
print(cfg.experiment_name)


class ItemSeqEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads=2, num_layers=2, max_len=20, causal=False):
        super().__init__()
        self.project_q = nn.Linear(emb_dim, emb_dim)
        self.project_k = nn.Linear(emb_dim, emb_dim)
        self.project_v = nn.Linear(emb_dim, emb_dim)
        self.transformer_encoder = TransformerEncoder(emb_dim, num_layers=num_layers, num_heads=num_heads, d_ff=emb_dim*4)
        self.position_encoder = LearnablePositionalEmbedding(emb_dim, max_len=max_len, init_std=1e-4)
        self.causal = causal

    def _build_attn_mask(self, seq_len, max_len, device):
        positions = torch.arange(max_len-1, -1, -1, device=device).unsqueeze(0)  # 1 L
        valid_mask = positions < seq_len.unsqueeze(1)  # B L
        key_padding_mask = ~valid_mask  # B L, True means masked
        attn_mask = key_padding_mask.unsqueeze(1).expand(-1, max_len, -1)  # B L L

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(max_len, max_len, dtype=torch.bool, device=device),
                diagonal=1
            )
            attn_mask = attn_mask | causal_mask.unsqueeze(0)

        return attn_mask, valid_mask

    def forward(self, seq, seq_len):
        """
        seq: [batch_size, max_len, emb_dim]
        seq_len: [batch_size] 记录每个样本的真实长度
        """
        batch_size, max_len, _ = seq.shape
        device = seq.device
        attn_mask, valid_mask = self._build_attn_mask(seq_len, max_len, device)
        positioned_seq = self.position_encoder.forward(seq) # 过大的绝对编码会淹没初始化值极小的Embedding，我正考虑怎么使用Repe替换之
        positioned_seq = positioned_seq * valid_mask.unsqueeze(-1)
        transformer_out = self.transformer_encoder.forward(
            positioned_seq,
            mask=attn_mask
        )
        transformer_out = transformer_out * valid_mask.unsqueeze(-1)
        batch_idx = torch.arange(batch_size, device=device)
        final_output = transformer_out[batch_idx, max_len-1, :]
        return final_output



class SpecialModel(BaseModel):
    def __init__(self, schema_manager: SchemaManager,):
        super(SpecialModel, self).__init__(schema_manager)
        self.manager = schema_manager
        self.LABEL = schema_manager.label_field
        self.DOMAIN = schema_manager.domain_field

        seq_dim = self.omni_embedding.seq_groups["history"].embedding_dim
        self.seq_dim = seq_dim
        # self.seq_encoder = AttentionSequencePoolingLayer(embedding_dim=seq_dim, att_hidden_units=(128, seq_dim))
        self.seq_encoder = ItemSeqEncoder(seq_dim, max_len=20+1, causal=False)
        # self.seq_encoder = SequencePoolingLayer()


        self.whole_input_dim = self.omni_embedding.whole.embedding_dim
        self.share_input_dim = schema_manager.source2emb_dim(FeatureSource.USER, FeatureSource.ITEM, FeatureSource.INTERACTION)
        self.specific_input_dim = schema_manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.ITEM_ID)
        self.item_id_input_dim = schema_manager.source2emb_dim(FeatureSource.ITEM_ID)
        self.user_id_input_dim = schema_manager.source2emb_dim(FeatureSource.USER_ID)

        self.share_expert = BifurcatedMLP(self.share_input_dim, self.share_input_dim // 2, 64, 1)
        self.specific_expert = BifurcatedMLP(self.specific_input_dim, 64, 1)
        self.user_id_expert = BifurcatedMLP(self.user_id_input_dim, 64, 1)
        self.item_id_expert = BifurcatedMLP(self.item_id_input_dim, 64, 1)
        self.seq_expert = BifurcatedMLP(seq_dim, 64, 1)
        self.add_expert = BifurcatedMLP(self.item_id_input_dim, 64, 1)

        # self.batch_norm = BatchNorm1d(6)
        self.gate_expert = MLP(self.specific_input_dim, 64, 6)

        self.gate_recorder = ExplicitFeatureRecorder()
        self._train_gate = True
        self._last_gates = None
        self.bifo = FeatureBifurcator(num_features=1)
        self.tmp_loss = None

    def train_gate(self):
        self._train_gate = True
    def freeze_gate(self):
        self._train_gate = False

    def _drop_none(self, list):
        return [emb for emb in list if emb is not None]

    def concat_embed_input_fields(self, interaction):
        user_emb_dict = self.omni_embedding.user_all.forward(interaction, split_by="source")
        user_id_emb = user_emb_dict[FeatureSource.USER_ID]
        user_side_emb = user_emb_dict.get(FeatureSource.USER, None)

        item_emb_dict = self.omni_embedding.item_all.forward(interaction, split_by="source")
        item_id_emb = item_emb_dict[FeatureSource.ITEM_ID]
        item_side_emb = item_emb_dict.get(FeatureSource.ITEM, None)

        inter_emb = self.omni_embedding.inter.forward(interaction)

        seq_emb, seq_tar, seq_len = self.omni_embedding.seq_groups["history"].fetch_all(interaction)
        share_emb = torch.cat(self._drop_none([user_side_emb, item_side_emb, inter_emb]), dim=-1)
        specific_emb = torch.cat(self._drop_none([user_id_emb, item_id_emb]), dim=-1)
        return seq_emb, share_emb, specific_emb, user_id_emb, item_id_emb, seq_tar, seq_len

    def forward(self, seq, sha, spe, uid, iid, seq_tar, seq_len):
        extended_seq = torch.cat([seq, seq_tar.unsqueeze(-2)], dim=-2)
        batch_size = seq.shape[0]
        batch_idx = torch.arange(batch_size, device=seq.device)
        flat_target_idx = seq_len.view(-1)
        extended_seq[batch_idx, flat_target_idx, :] = extended_seq[batch_idx, -1, :]
        final_seq = extended_seq

        _, uid_logits = self.user_id_expert(uid)
        _, iid_logits = self.item_id_expert(iid)
        _, spe_logits = self.specific_expert(spe)
        _, sha_logits = self.share_expert(sha)
        _, seq_logits = self.seq_expert(self.seq_encoder.forward(final_seq, seq_len+1))
        _, add_logits = self.add_expert(uid + iid)
        gates = self.gate_expert(spe)
        if not self._train_gate:
            gates = gates.detach()

        logits = [uid_logits, iid_logits, spe_logits, seq_logits, sha_logits, add_logits]
        all_logits = torch.cat(logits, dim=-1)

        self._last_uid_logits = uid_logits
        self._last_iid_logits = iid_logits
        self._last_sha_logits = sha_logits

        # 选择mask掉谁
        mask = torch.zeros_like(gates, dtype=torch.bool)
        mask[:, -2:] = True
        gates = gates.masked_fill(mask, float('-inf'))
        # 结束

        gates = F.softmax(gates, dim=-1)
        self.gate_recorder.record("three_expert", gates)
        self.gate_recorder.record("ctr_contribute", gates * all_logits)
        self.gate_recorder.record("bias", self.bifo.bias)
        bias, final_logits = self.bifo(torch.sum(gates * all_logits, dim=-1))  # B
        final_logits += bias
        self._last_gates = gates
        return final_logits

    def predict(self, interaction):
        who, sha, spe, uid, iid, tar, seq_len = self.concat_embed_input_fields(interaction)
        return self.forward(who, sha, spe, uid, iid, tar, seq_len)

    def _calc_correlation_penalty(self, x, y):
        """
        计算两个 logit 分布的皮尔逊相关系数惩罚
        x, y shape: [Batch_size, 1] 或 [Batch_size]
        """
        # 1. 去均值 (Centered)
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)

        # 2. 计算相似度 (Pearson Correlation r)
        # 添加 eps=1e-8 防止方差为0时除以0导致 NaN
        r = F.cosine_similarity(x_centered, y_centered, dim=0, eps=1e-8)

        # 3. 计算惩罚
        # 使用 ReLU 只惩罚正相关（同增同减），允许负相关（互补）和零相关
        # 如果你希望它们完全独立（既不同增，也不同减），可以改成求绝对值或平方：r.abs().mean()
        return F.relu(r).mean()

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        final_logits = self.predict(interaction)

        # 原有的 Gate Loss
        gate_loss = -0.02 * torch.var(self._last_gates, dim=0).mean()

        # --- 新增：专家去相关性惩罚 (Decorrelation Loss) ---
        corr_sha_uid = self._calc_correlation_penalty(self._last_sha_logits, self._last_uid_logits)
        corr_sha_iid = self._calc_correlation_penalty(self._last_sha_logits, self._last_iid_logits)

        # 这个权重 (0.05) 是一个超参，需要根据你的模型 scale 微调。
        # 太小没效果，太大可能会干扰正常的交叉熵训练。建议从 0.01 到 0.1 之间尝试。
        alpha_decorr = 0.03
        decorr_loss = alpha_decorr * (corr_sha_uid + corr_sha_iid)
        mean_gates = self._last_gates[:, :4].mean(dim=0)
        importance_loss = 0.02 * torch.var(mean_gates)
        # ----------------------------------------------------

        # 总 Loss 合并
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels) + gate_loss + decorr_loss + importance_loss
        return loss



if __name__ == '__main__':
    from betterbole.utils.task_chain import auto_queue
    auto_queue()
    device = "cuda"
    history_group = SeqGroupConfig(
        group_name="history",
        seq_len_field_name="seq_len",
        max_len=20,
        padding_side='right'
    )
    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, cfg.emb_dim, min_freq=10,use_oov=True) # 显存太低没法拉高dim
    item_setting = SparseEmbSetting("movie_id", FeatureSource.ITEM_ID, cfg.emb_dim, min_freq=10, use_oov=True) # 显存太低没法拉高dim
    genres_setting = SparseSetEmbSetting("genres", FeatureSource.ITEM, 8, min_freq=10, use_oov=True)
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("age", FeatureSource.USER, 8),
        SparseEmbSetting("gender", FeatureSource.USER, 8),
        SparseEmbSetting("occupation", FeatureSource.USER, 8),
        genres_setting,
        SharedVocabSeqSetting("movie_id_seq", item_setting, group=history_group),
        SharedVocabSeqSetting("genres_seq", genres_setting, group=history_group)
    ]

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
    # 序列建模
    whole_lf = extract_history_sequences(whole_lf, max_seq_len=20, user_col="user_id", time_col="timestamp", feature_mapping={
        "movie_id": "movie_id_seq",
        "genres": "genres_seq"
    }, label_col="label", seq_len_col="seq_len")
    print(whole_lf.head(5).collect())

    manager = SchemaManager(settings_list, preset_workdir(cfg.dataset_name), time_field="timestamp", label_fields="label", domain_fields="gender")
    print(manager.fields())
    train_raw, valid_raw, test_raw = manager.split_dataset(whole_lf, strategy="sequential_ratio")
    manager.fit(train_raw, low_memory=True)
    train_lf = manager.transform(train_raw)
    valid_lf = manager.transform(valid_raw)
    test_lf = manager.transform(test_raw)

    print(train_lf.head(5).collect())

    train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf, test_lf)
    print(pl.scan_parquet(train_path).collect_schema().names())
    print("架构编译成功，可供调用。")
    evaluator = LogDecorator(Evaluator("AUC"),
                             save_path=manager.work_dir / "logs.log", title=f"DIM={cfg.emb_dim},{cfg.experiment_name}")

    ps_dataset = ParquetStreamDataset(train_path, manager) # 更少的读取
    ps_valid = ParquetStreamDataset(valid_path, manager, batch_size=4096 * 2) # 更少的读取
    ps_dataloader = DataLoader(
        ps_dataset,
        batch_size=None,  # 必须为 None，让 Dataset 自己划分 Batch
        num_workers=0,  # 开启 2 个进程并行读取
        pin_memory=True  # 测试时不占用显存，如果是真实 GPU 训练设为 True
    )

    model = SpecialModel(manager).to(device)


    from betterbole.utils.time import CudaNamedTimer
    ntr = CudaNamedTimer()
    named_parameters = split_params_by_decay(model.named_parameters(), weight_decay=1e-5, no_decay_keywords=[]) # 对Embedding施加wd反而更好
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
        print(model.gate_recorder.get_window_stats())
        print(f"Validation Metrics: {metrics_result}")
        evaluator.clear()
        ntr.report()
        # print(f"=== Epoch {epoch} Done, Average Loss: {total_loss / batch_count:.4f} ===")
