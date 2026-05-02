from dataclasses import dataclass
from typing import Optional, Tuple

import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.enum_type import FeatureSource
from betterbole.data.dataset import ParquetStreamDataset
from betterbole.datasets.movielens import MovieLensDataset
from betterbole.emb import SchemaManager
from betterbole.emb.schema import (
    MultiSparseSetting,
    SeqGroupConfig,
    SequenceSetting,
    SparseEmbSetting,
)
from betterbole.evaluate.evaluator import Evaluator, LogDecorator
from betterbole.experiment import change_root_workdir, preset_workdir
from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.models.base import BaseModel
from betterbole.models.utils.general import BifurcatedMLP, FeatureBifurcator, MLP
from betterbole.utils import RelativeTimeEmbedding
from betterbole.utils.optimize import split_params_by_decay
from betterbole.utils.recorder import ExplicitFeatureRecorder
from betterbole.utils.sequential import extract_history_sequences

change_root_workdir()


@dataclass
class MovieLensConfig(ConfigBase):
    seed: str = 2025
    experiment_name: str = "ml1m-hyformer-sigmoid"
    dataset_name: str = "movie-lens-hyformer"
    emb_dim: int = 16
    seq_dim: int = 16
    history_max_len: int = 20
    num_queries: int = 2
    num_hyformer_blocks: int = 2
    num_heads: int = 4
    dropout_rate: float = 0.1
    query_aux_loss_weight: float = 0.2


cfg: MovieLensConfig = ParamManager(MovieLensConfig).build()
print(cfg.dataset_name)
print(cfg.experiment_name)


def build_left_padded_mask(seq_len: torch.Tensor, max_len: int, device: torch.device):
    positions = torch.arange(max_len - 1, -1, -1, device=device).unsqueeze(0)
    valid_mask = positions < seq_len.unsqueeze(1)
    padding_mask = ~valid_mask
    return padding_mask, valid_mask


class TokenProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.net = None
        if input_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

    def forward(
        self,
        x: Optional[torch.Tensor],
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self.net is None or x is None:
            if batch_size is None or device is None:
                raise ValueError("batch_size and device are required when projector input is None.")
            return torch.zeros(batch_size, self.output_dim, device=device)
        return F.silu(self.net(x))


class StaticFeatureScaler(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(input_dim)) if input_dim > 0 else None

    def forward(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or self.gate_logits is None:
            return x
        gate = torch.sigmoid(self.gate_logits).to(dtype=x.dtype)
        return x * gate


class QueryGenerator(nn.Module):
    def __init__(self, d_model: int, num_ns_tokens: int, num_queries: int):
        super().__init__()
        self.num_queries = num_queries
        global_info_dim = num_ns_tokens * d_model
        self.global_norm = nn.LayerNorm(global_info_dim)
        self.query_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * 4),
                    nn.SiLU(),
                    nn.Linear(d_model * 4, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ]
        )

    def forward(
        self,
        ns_tokens: torch.Tensor,
    ) -> torch.Tensor:
        global_info = ns_tokens.reshape(ns_tokens.shape[0], -1)
        global_info = self.global_norm(global_info)
        queries = [ffn(global_info) for ffn in self.query_ffns]
        return torch.stack(queries, dim=1)


class SequenceEvolutionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, seq_tokens: torch.Tensor, seq_padding_mask: torch.Tensor) -> torch.Tensor:
        residual = seq_tokens
        seq_norm = self.norm1(seq_tokens)
        attn_out, _ = self.self_attn(
            seq_norm,
            seq_norm,
            seq_norm,
            key_padding_mask=seq_padding_mask,
            need_weights=False,
        )
        attn_out = torch.nan_to_num(attn_out)
        seq_tokens = residual + attn_out
        seq_tokens = seq_tokens.masked_fill(seq_padding_mask.unsqueeze(-1), 0.0)

        residual = seq_tokens
        seq_tokens = residual + self.ffn(self.norm2(seq_tokens))
        seq_tokens = seq_tokens.masked_fill(seq_padding_mask.unsqueeze(-1), 0.0)
        return seq_tokens


class QueryDecodingBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = query_tokens
        q_norm = self.norm_q(query_tokens)
        kv_norm = self.norm_kv(seq_tokens)
        attn_out, _ = self.cross_attn(
            q_norm,
            kv_norm,
            kv_norm,
            key_padding_mask=seq_padding_mask,
            need_weights=False,
        )
        attn_out = torch.nan_to_num(attn_out)
        return residual + attn_out


class RankMixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_total: int,
        dropout_rate: float,
        mode: str = "full",
    ):
        super().__init__()
        self.mode = mode
        self.n_total = n_total

        if mode == "full":
            if d_model % n_total != 0:
                raise ValueError(f"d_model={d_model} must be divisible by n_total={n_total}.")
            self.d_sub = d_model // n_total

        if mode != "none":
            self.norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model * 4, d_model),
            )
            self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, d_model = x.shape
        x = x.view(batch_size, token_count, self.n_total, self.d_sub)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, token_count, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x

        mixed = self.token_mixing(x) if self.mode == "full" else x
        boosted = self.ffn(self.norm(mixed))
        return self.post_norm(x + boosted)


class HyFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns_tokens: int,
        dropout_rate: float,
        rank_mixer_mode: str = "full",
    ):
        super().__init__()
        self.num_queries = num_queries
        self.seq_encoder = SequenceEvolutionBlock(d_model, num_heads, dropout_rate)
        self.query_decoder = QueryDecodingBlock(d_model, num_heads, dropout_rate)
        self.rank_mixer = RankMixerBlock(
            d_model=d_model,
            n_total=num_queries + num_ns_tokens,
            dropout_rate=dropout_rate,
            mode=rank_mixer_mode,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_padding_mask: torch.Tensor,
    ):
        next_seq = self.seq_encoder(seq_tokens, seq_padding_mask) # 平平无奇的序列建模，甚至你不把目标信息加上就做建模
        decoded_query = self.query_decoder(query_tokens, next_seq, seq_padding_mask) # 单层输出，同时只拿query要的东西，实在不太美观 疑点2 他们都是使用单层注意力进行建模吗，为什么不至少双层
        combined_tokens = torch.cat([decoded_query, ns_tokens], dim=1)
        combined_tokens = self.rank_mixer(combined_tokens) # 单层Mix，Mix只做单层有表达能力吗 疑点3 他们怎么都是单层
        next_query = combined_tokens[:, : self.num_queries, :]
        next_ns = combined_tokens[:, self.num_queries :, :]
        return next_query, next_ns, next_seq


class HyFormerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_ns_tokens: int,
        num_queries: int,
        num_hyformer_blocks: int,
        num_heads: int,
        dropout_rate: float,
        rank_mixer_mode: str = "full",
    ):
        super().__init__()
        self.query_generator = QueryGenerator(d_model, num_ns_tokens, num_queries)
        self.blocks = nn.ModuleList(
            [
                HyFormerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_queries=num_queries,
                    num_ns_tokens=num_ns_tokens,
                    dropout_rate=dropout_rate,
                    rank_mixer_mode=rank_mixer_mode,
                )
                for _ in range(num_hyformer_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padding_mask, _ = build_left_padded_mask(seq_len, seq_tokens.shape[1], seq_tokens.device)
        query_tokens = self.query_generator(ns_tokens)
        query_seed_tokens = query_tokens
        query_seed = query_tokens.reshape(query_tokens.shape[0], -1)

        query_tokens = self.dropout(query_tokens)
        ns_tokens = self.dropout(ns_tokens)
        seq_tokens = self.dropout(seq_tokens)

        for block in self.blocks:
            query_tokens, ns_tokens, seq_tokens = block(
                query_tokens,
                ns_tokens,
                seq_tokens,
                padding_mask,
            )

        return self.output_proj(query_tokens.reshape(query_tokens.shape[0], -1)), query_seed, query_seed_tokens


class SpecialModel(BaseModel):
    def __init__(self, schema_manager: SchemaManager):
        super().__init__(schema_manager)
        self.manager = schema_manager
        self.LABEL = schema_manager.label_field
        self.DOMAIN = schema_manager.domain_field

        self.history_view = self.omni_embedding.seq_groups["history"]
        self.history_seq_len_field = "seq_len"
        self.history_time_field = "action_timestamp"

        self.raw_seq_dim = self.history_view.embedding_dim
        self.seq_dim = cfg.seq_dim
        self.num_ns_tokens = 2
        self.seq_step_project = nn.Linear(self.raw_seq_dim, self.seq_dim)
        self.real_time = RelativeTimeEmbedding(embedding_dim=self.seq_dim)

        self.share_input_dim = schema_manager.source2emb_dim(
            FeatureSource.USER,
            FeatureSource.ITEM,
        )
        self.specific_input_dim = schema_manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.ITEM_ID)
        self.item_id_input_dim = schema_manager.source2emb_dim(FeatureSource.ITEM_ID)
        self.user_id_input_dim = schema_manager.source2emb_dim(FeatureSource.USER_ID)
        self.user_side_input_dim = schema_manager.source2emb_dim(FeatureSource.USER)
        self.item_side_input_dim = schema_manager.source2emb_dim(FeatureSource.ITEM)
        self.user_group_input_dim = self.user_id_input_dim + self.user_side_input_dim
        self.item_group_input_dim = self.item_id_input_dim + self.item_side_input_dim

        self.user_id_scale = StaticFeatureScaler(self.user_id_input_dim)
        self.user_side_scale = StaticFeatureScaler(self.user_side_input_dim)
        self.item_id_scale = StaticFeatureScaler(self.item_id_input_dim)
        self.item_side_scale = StaticFeatureScaler(self.item_side_input_dim)

        self.user_group_token = TokenProjector(self.user_group_input_dim, self.seq_dim)
        self.item_group_token = TokenProjector(self.item_group_input_dim, self.seq_dim)

        self.hyformer = HyFormerEncoder(
            d_model=self.seq_dim,
            num_ns_tokens=self.num_ns_tokens,
            num_queries=cfg.num_queries,
            num_hyformer_blocks=cfg.num_hyformer_blocks,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
            rank_mixer_mode="full",
        )
        self.share_expert = BifurcatedMLP(self.share_input_dim, self.share_input_dim // 2, 64, 1)
        self.specific_expert = BifurcatedMLP(self.specific_input_dim, 64, 1)
        self.user_id_expert = BifurcatedMLP(self.user_id_input_dim, 64, 1)
        self.item_id_expert = BifurcatedMLP(self.item_id_input_dim, 64, 1)
        self.seq_expert = BifurcatedMLP(self.seq_dim, 64, 1)
        self.add_expert = BifurcatedMLP(self.item_id_input_dim, 64, 1)
        self.query_seed_head = MLP(cfg.num_queries * self.seq_dim, 64, 1, dropout_rate=cfg.dropout_rate)

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

    def _drop_none(self, values):
        return [emb for emb in values if emb is not None]

    def _zeros(self, batch_size: int, dim: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, dim, device=device)

    def build_anonymous_sequence(self, interaction):
        hist_seq = self.history_view.forward(interaction, split_by="none")
        hist_seq = self.seq_step_project(hist_seq)

        seq_len = interaction[self.history_seq_len_field]
        curr_ts = interaction[self.manager.time_field]
        hist_timestamp = interaction[self.history_time_field]
        hist_seq = hist_seq + self.real_time(curr_ts, hist_timestamp, seq_lens=seq_len)

        empty_mask = seq_len == 0
        effective_seq_len = seq_len.clone()
        if empty_mask.any():
            hist_seq = hist_seq.clone()
            hist_seq[empty_mask, -1, :] = 0.0
            effective_seq_len[empty_mask] = 1

        return hist_seq, effective_seq_len

    def build_ns_tokens(self, user_id_emb, user_side_emb, item_id_emb, item_side_emb):
        batch_size = user_id_emb.shape[0]
        device = user_id_emb.device
        user_side = user_side_emb if user_side_emb is not None else self._zeros(batch_size, self.user_side_input_dim, device)
        item_side = item_side_emb if item_side_emb is not None else self._zeros(batch_size, self.item_side_input_dim, device)
        user_group = torch.cat([user_id_emb, user_side], dim=-1)
        item_group = torch.cat([item_id_emb, item_side], dim=-1)
        ns_tokens = [
            self.user_group_token(user_group, batch_size, device),
            self.item_group_token(item_group, batch_size, device),
        ] # 疑点1：这里为什么要这么分散地聚合各个token们？看上去仅仅只是为了让他们Embedding维度相同，为什么不直接whole cat起来生成一个全局查询token
        return torch.stack(ns_tokens, dim=1)

    def concat_embed_input_fields(self, interaction):
        user_emb_dict = self.omni_embedding.user_all.forward(interaction, split_by="source")
        user_id_emb = self.user_id_scale(user_emb_dict[FeatureSource.USER_ID])
        user_side_emb = self.user_side_scale(user_emb_dict.get(FeatureSource.USER, None))

        item_emb_dict = self.omni_embedding.item_all.forward(interaction, split_by="source")
        item_id_emb = self.item_id_scale(item_emb_dict[FeatureSource.ITEM_ID])
        item_side_emb = self.item_side_scale(item_emb_dict.get(FeatureSource.ITEM, None))

        anon_seq, anon_seq_len = self.build_anonymous_sequence(interaction)
        ns_tokens = self.build_ns_tokens(
            user_id_emb,
            user_side_emb,
            item_id_emb,
            item_side_emb,
        )

        share_emb = torch.cat(self._drop_none([user_side_emb, item_side_emb]), dim=-1)
        specific_emb = torch.cat(self._drop_none([user_id_emb, item_id_emb]), dim=-1)
        return anon_seq, ns_tokens, share_emb, specific_emb, user_id_emb, item_id_emb, anon_seq_len

    def forward(self, anon_seq, ns_tokens, sha, spe, uid, iid, anon_seq_len):
        hyformer_output, query_seed, _ = self.hyformer(ns_tokens, anon_seq, anon_seq_len)

        _, uid_logits = self.user_id_expert(uid)
        _, iid_logits = self.item_id_expert(iid)
        _, spe_logits = self.specific_expert(spe)
        _, sha_logits = self.share_expert(sha)
        _, seq_logits = self.seq_expert(hyformer_output)
        _, add_logits = self.add_expert(uid + iid)
        query_seed_logits = self.query_seed_head(query_seed)
        gates = self.gate_expert(spe)
        if not self._train_gate:
            gates = gates.detach()

        logits = [uid_logits, iid_logits, spe_logits, seq_logits, sha_logits, add_logits]
        all_logits = torch.cat(logits, dim=-1)

        self._last_uid_logits = uid_logits
        self._last_iid_logits = iid_logits
        self._last_sha_logits = sha_logits
        self._last_query_seed_logits = query_seed_logits

        mask = torch.zeros_like(gates, dtype=torch.bool)
        mask[:, -2:] = True
        gates = gates.masked_fill(mask, float("-inf"))

        gates = F.softmax(gates, dim=-1)
        self.gate_recorder.record("three_expert", gates)
        self.gate_recorder.record("ctr_contribute", gates * all_logits)
        self.gate_recorder.record("bias", self.bifo.bias)
        bias, final_logits = self.bifo(torch.sum(gates * all_logits, dim=-1))
        final_logits += bias
        self._last_gates = gates
        return final_logits

    def predict(self, interaction):
        anon_seq, ns_tokens, sha, spe, uid, iid, anon_seq_len = self.concat_embed_input_fields(interaction)
        return self.forward(anon_seq, ns_tokens, sha, spe, uid, iid, anon_seq_len)

    def _calc_correlation_penalty(self, x, y):
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        r = F.cosine_similarity(x_centered, y_centered, dim=0, eps=1e-8)
        return F.relu(r).mean()

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        final_logits = self.predict(interaction)

        gate_loss = -0.02 * torch.var(self._last_gates, dim=0).mean()
        corr_sha_uid = self._calc_correlation_penalty(self._last_sha_logits, self._last_uid_logits)
        corr_sha_iid = self._calc_correlation_penalty(self._last_sha_logits, self._last_iid_logits)
        alpha_decorr = 0.03
        decorr_loss = alpha_decorr * (corr_sha_uid + corr_sha_iid)
        mean_gates = self._last_gates[:, :4].mean(dim=0)
        importance_loss = 0.02 * torch.var(mean_gates)
        aux_query_loss = cfg.query_aux_loss_weight * nn.functional.binary_cross_entropy_with_logits(
            self._last_query_seed_logits.squeeze(-1), labels
        )
        loss = (
            nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
            + gate_loss
            + decorr_loss
            + importance_loss
            + aux_query_loss
        )
        return loss


if __name__ == "__main__":
    from betterbole.utils.task_chain import auto_queue
    from betterbole.utils.time import CudaNamedTimer

    auto_queue()
    device = "cuda"

    history_group = SeqGroupConfig(
        group_name="history",
        seq_len_field_name="seq_len",
        max_len=cfg.history_max_len,
        padding_side="left",
        time_field_name="action_timestamp",
    )

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, cfg.emb_dim, min_freq=10, use_oov=True)
    item_setting = SparseEmbSetting("movie_id", FeatureSource.ITEM_ID, cfg.emb_dim, min_freq=10, use_oov=True)
    genres_setting = MultiSparseSetting(
        "genres",
        FeatureSource.ITEM,
        embedding_dim=16,
        max_tag_len=8,
        min_freq=10,
        use_oov=True,
    )

    history_item_element = SparseEmbSetting(
        "movie_id_seq_token",
        FeatureSource.SEQ,
        embedding_dim=16,
        min_freq=10,
        use_oov=True,
    )
    history_genres_element = MultiSparseSetting(
        "genres_seq_token",
        FeatureSource.SEQ,
        embedding_dim=16,
        max_tag_len=8,
        min_freq=10,
        use_oov=True,
    )

    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("age", FeatureSource.USER, 16),
        SparseEmbSetting("gender", FeatureSource.USER, 16),
        SparseEmbSetting("occupation", FeatureSource.USER, 16),
        genres_setting,
        SequenceSetting("movie_id_seq", element_setting=history_item_element, group=history_group),
        SequenceSetting("genres_seq", element_setting=history_genres_element, group=history_group),
    ]

    user_lf = pl.from_pandas(MovieLensDataset.USER_FEATURES_DF).lazy()
    item_lf = pl.from_pandas(MovieLensDataset.ITEM_FEATURES_DF).lazy()
    inter_lf = pl.from_pandas(MovieLensDataset.INTERACTION_DF).lazy()
    whole_lf: pl.LazyFrame = inter_lf.join(item_lf, on="movie_id", how="left").join(user_lf, on="user_id", how="left")
    whole_lf = whole_lf.with_columns(
        pl.col("genres").str.split("|"),
        (pl.col("rating") >= 4).cast(pl.Int8).alias("label"),
    )
    whole_lf = whole_lf.with_columns(
        pl.when(pl.col("age") < 25)
        .then(0)
        .when(pl.col("age") < 35)
        .then(1)
        .otherwise(2)
        .alias("domain_id")
    )

    whole_lf = extract_history_sequences(
        whole_lf,
        max_seq_len=cfg.history_max_len,
        user_col="user_id",
        time_col="timestamp",
        feature_mapping={
            "movie_id": "movie_id_seq",
            "genres": "genres_seq",
            "timestamp": "action_timestamp",
        },
        label_col="label",
        seq_len_col="seq_len",
    )
    print(whole_lf.head(5).collect())

    manager = SchemaManager(
        settings_list,
        preset_workdir(cfg.dataset_name),
        time_field="timestamp",
        label_fields="label",
        domain_fields="gender",
    )
    print(manager.fields())
    train_raw, valid_raw, test_raw = manager.split_dataset(whole_lf, strategy="sequential_ratio")
    manager.fit(train_raw, low_memory=True)
    train_lf = manager.transform(train_raw)
    valid_lf = manager.transform(valid_raw)
    test_lf = manager.transform(test_raw)

    print(train_lf.head(5).collect())

    train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf, test_lf)
    print(pl.scan_parquet(train_path).collect_schema().names())
    print("HyFormer 匿名序列架构编译成功，可供调用。")

    evaluator = LogDecorator(
        Evaluator("AUC"),
        save_path=manager.work_dir / "logs.log",
        title=f"DIM={cfg.emb_dim},{cfg.experiment_name}",
    )

    ps_dataset = ParquetStreamDataset(train_path, manager)
    ps_valid = ParquetStreamDataset(valid_path, manager, batch_size=4096 * 2)

    model = SpecialModel(manager).to(device)

    ntr = CudaNamedTimer()
    named_parameters = split_params_by_decay(model.named_parameters(), weight_decay=1e-5, no_decay_keywords=[])
    optimizer = torch.optim.Adam(named_parameters, lr=1e-3)
    for epoch in range(50):
        batch_count = 0
        model.train()
        with ntr("epoch"):
            for batch_interaction in ps_dataset:
                with ntr("prepare"):
                    batch_interaction = batch_interaction.to(device)

                with ntr("train"):
                    optimizer.zero_grad()
                    loss = model.calculate_loss(batch_interaction)
                    loss.backward()
                    optimizer.step()

                batch_count += 1
                if batch_count % 100 == 0:
                    ntr.report()
                    print(f"Epoch {epoch}, Batch {batch_count}, Current Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            for batch_interaction in ps_valid:
                batch_interaction = batch_interaction.to(device)
                uids = batch_interaction[manager.uid_field]
                labels = batch_interaction[manager.label_field]
                scores = model.predict(batch_interaction)
                evaluator.collect_pointwise(uids, labels, batch_preds_1d=scores)

        metrics_result = evaluator.summary(epoch)
        print(model.gate_recorder.get_window_stats())
        print(f"Validation Metrics: {metrics_result}")
        evaluator.clear()
        ntr.report()
