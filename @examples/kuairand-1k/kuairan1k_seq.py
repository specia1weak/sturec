import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import polars as pl
import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource
from betterbole.core.train import EarlyStopper
from betterbole.core.train.context import TrainerComponents, TrainerDataLoaders
from betterbole.core.train.trainer import BaseTrainer
from betterbole.data.dataset import ParquetStreamDataset
from betterbole.datasets.kuairand import KuaiRandDataset
from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import EmbView
from betterbole.emb.schema import MultiSparseSetting, SeqGroupConfig, SequenceSetting, SparseEmbSetting
from betterbole.evaluate.evaluator import Evaluator
from betterbole.evaluate.manager import DomainFilter, EvaluatorManager
from betterbole.experiment import WORKSPACE, change_root_workdir
from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components.heads import DomainTowerHead
from betterbole.models.utils.common import build_mlp, to_dims
from betterbole.models.utils.sequence import AttentionSequencePoolingLayer
from betterbole.utils.optimize import split_params_by_decay
from betterbole.utils.sequential import extract_history_sequences

change_root_workdir()


@dataclass
class KuairandSeqConfig(ConfigBase):
    experiment_name: str = "din_sharedbottom"
    dataset_name: str = "kuairand-rand-seq-v2"
    seed: int = 2026
    device: str = "cuda"
    max_epochs: int = 3
    ckpt_dir: str = ""

    batch_size: int = 4096
    eval_batch_size: int = 8192
    id_emb: int = 32
    side_emb: int = 32
    seq_max_len: int = 20
    top_k_domains: int = 5
    min_freq: int = 10
    model: str = "din_sharedbottom"
    log_name: str = "seq_sharedbottom.log"
    weight_decay: float = 1e-6
    learning_rate: float = 1e-3
    prepare_only: bool = False
    redo_cache: bool = False


MODEL_KWARGS: Dict[str, Dict[str, object]] = {
    "din_sharedbottom": {
        "bottom_hidden_dims": (512, 256, 128),
        "dropout_rate": 0.1,
        "batch_norm": True,
        "tower_hidden_dims": (128,),
        "tower_dropout_rate": 0.2,
        "att_hidden_units": (128, 32),
    }
}

pm = ParamManager(KuairandSeqConfig)
cfg: KuairandSeqConfig = pm.build()

print(cfg)
time.sleep(1)


def build_settings(cfg: KuairandSeqConfig):
    user_setting = SparseEmbSetting(
        "user_id",
        FeatureSource.USER_ID,
        cfg.id_emb,
        min_freq=cfg.min_freq,
        use_oov=True,
    )
    item_setting = SparseEmbSetting(
        "video_id",
        FeatureSource.ITEM_ID,
        cfg.id_emb,
        min_freq=cfg.min_freq,
        use_oov=True,
    )
    tab_setting = SparseEmbSetting(
        "tab",
        FeatureSource.INTERACTION,
        cfg.side_emb,
        padding_zero=True,
        use_null=True,
        use_oov=True,
        min_freq=cfg.min_freq,
    )

    user_side_settings = [
        SparseEmbSetting("user_active_degree", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("is_live_streamer", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("is_video_author", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("friend_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("register_days_range", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("follow_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("fans_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("is_lowactive_period", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat0", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat1", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat2", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat3", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat4", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat5", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat6", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat7", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat8", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat9", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat10", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat11", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat12", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat13", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat14", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat15", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat16", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("onehot_feat17", FeatureSource.USER, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
    ]

    item_side_settings = [
        SparseEmbSetting("author_id", FeatureSource.ITEM, cfg.id_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("music_id", FeatureSource.ITEM, cfg.id_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("video_type", FeatureSource.ITEM, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("music_type", FeatureSource.ITEM, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
        SparseEmbSetting("visible_status", FeatureSource.ITEM, cfg.side_emb, min_freq=cfg.min_freq, use_oov=True),
    ]

    tag_setting = MultiSparseSetting(
        "tag",
        FeatureSource.ITEM,
        cfg.side_emb,
        max_tag_len=5,
        is_string_format=True,
        separator=",",
        min_freq=cfg.min_freq,
        use_oov=True,
    )

    seq_group = SeqGroupConfig(
        group_name="user_history",
        seq_len_field_name="seq_len",
        max_len=cfg.seq_max_len,
        padding_side="right",
        time_field_name="time_ms_seq",
    )

    history_settings = [
        SequenceSetting("video_id_seq", item_setting, seq_group),
        SequenceSetting("tab_seq", tab_setting, seq_group),
        SequenceSetting("author_id_seq", item_side_settings[0], seq_group),
        SequenceSetting("music_id_seq", item_side_settings[1], seq_group),
        SequenceSetting("video_type_seq", item_side_settings[2], seq_group),
        SequenceSetting("music_type_seq", item_side_settings[3], seq_group),
        SequenceSetting("visible_status_seq", item_side_settings[4], seq_group),
        SequenceSetting("tag_seq", tag_setting, seq_group),
    ]

    return [
        user_setting,
        item_setting,
        tab_setting,
        *user_side_settings,
        *item_side_settings,
        tag_setting,
        *history_settings,
    ]


def build_manager(cfg: KuairandSeqConfig) -> SchemaManager:
    work_dir = WORKSPACE / cfg.dataset_name
    return SchemaManager(
        build_settings(cfg),
        work_dir,
        time_field="time_ms",
        label_fields="is_click",
        domain_fields="tab",
    )


def build_joined_frame(cfg: KuairandSeqConfig) -> pl.LazyFrame:
    user_lf = pl.scan_csv(KuaiRandDataset.USER_FEATURES)
    item_lf = pl.scan_csv(KuaiRandDataset.VIDEO_FEATURES)
    inter_lf = pl.scan_csv(
        [
            KuaiRandDataset.STD_LOG_FORMER_DATA_P1,
            KuaiRandDataset.STD_LOG_FORMER_DATA_P2,
            KuaiRandDataset.RAND_LOG_FORMER_DATA,
        ]
    )
    whole_lf = inter_lf.join(item_lf, on="video_id", how="left").join(user_lf, on="user_id", how="left")

    top_k_tabs = (
        whole_lf.group_by("tab")
        .len()
        .sort("len", descending=True)
        .head(cfg.top_k_domains)
        .select("tab")
        .collect()
        .get_column("tab")
        .to_list()
    )
    print("top tabs:", top_k_tabs)
    return whole_lf.filter(pl.col("tab").is_in(top_k_tabs))


def prepare_sequence_dataset(manager: SchemaManager, cfg: KuairandSeqConfig) -> Tuple[str, str, str]:
    work_dir = Path(manager.work_dir)
    whole_lf = build_joined_frame(cfg)
    whole_lf = manager.make_checkpoint(
        whole_lf,
        file_name="kuairand_seq_joined.parquet",
        redo=cfg.redo_cache,
    )

    history_lf = extract_history_sequences(
        whole_lf,
        max_seq_len=cfg.seq_max_len,
        user_col="user_id",
        time_col="time_ms",
        feature_mapping={
            "video_id": "video_id_seq",
            "tab": "tab_seq",
            "author_id": "author_id_seq",
            "music_id": "music_id_seq",
            "video_type": "video_type_seq",
            "music_type": "music_type_seq",
            "visible_status": "visible_status_seq",
            "tag": "tag_seq",
            "time_ms": "time_ms_seq",
        },
        seq_len_col="seq_len",
        label_col="is_click",
        positive_label=1,
    )

    train_raw = history_lf.filter(pl.col("date") <= 20220506)
    valid_raw = history_lf.filter(pl.col("date") == 20220507)
    test_raw = history_lf.filter(pl.col("date") == 20220508)

    manager.fit(train_raw, low_memory=True)

    selected_fields = manager.fields()
    train_lf = manager.transform(train_raw).select(selected_fields)
    valid_lf = manager.transform(valid_raw).select(selected_fields)
    test_lf = manager.transform(test_raw).select(selected_fields)

    train_lf = manager.make_checkpoint(
        train_lf,
        file_name="train.parquet",
        redo=cfg.redo_cache,
        sort_by="time_ms",
    )
    valid_lf = manager.make_checkpoint(
        valid_lf,
        file_name="valid.parquet",
        redo=cfg.redo_cache,
        sort_by="time_ms",
    )
    test_lf = manager.make_checkpoint(
        test_lf,
        file_name="test.parquet",
        redo=cfg.redo_cache,
        sort_by="time_ms",
    )

    print("schema fields:", selected_fields)
    print("train sample:")
    print(
        train_lf.select(
            [
                "time_ms",
                "user_id",
                "video_id_seq",
                "tab_seq",
                "time_ms_seq",
                "seq_len",
                "is_click",
                "tab",
            ]
        )
        .head(3)
        .collect()
    )

    train_path = str(work_dir / "train.parquet")
    valid_path = str(work_dir / "valid.parquet")
    test_path = str(work_dir / "test.parquet")
    print("sequence dataset ready:", train_path, valid_path, test_path)
    return train_path, valid_path, test_path


class DINSharedBottomModel(MSRModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        bottom_hidden_dims: Iterable[int] = (512, 256, 128),
        dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = True,
        tower_hidden_dims: Iterable[int] = (128,),
        tower_dropout_rate: float = 0.2,
        att_hidden_units: Iterable[int] = (128, 32),
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = manager.domain_field
        self.LABEL = manager.label_field

        base_fields = [
            setting.field_name
            for setting in manager.settings
            if not isinstance(setting, SequenceSetting) and setting.field_name != self.DOMAIN
        ]
        self.current_view = EmbView(self.omni_embedding, include_fields=base_fields)
        self.history_view = self.omni_embedding.seq_groups["user_history"]

        domain_setting = manager.get_setting(self.DOMAIN)
        self.domain_shift = 1 if getattr(domain_setting, "padding_zero", False) else 0

        hidden_dims = to_dims(bottom_hidden_dims, default=(512, 256, 128))
        self.bottom_input_dim = self.current_view.embedding_dim + self.history_view.embedding_dim
        self.bottom_output_dim = int(hidden_dims[-1])
        self.att_pool = AttentionSequencePoolingLayer(
            embedding_dim=self.history_view.embedding_dim,
            att_hidden_units=tuple(int(v) for v in att_hidden_units),
            att_activation="sigmoid",
        )
        self.bottom = build_mlp(
            self.bottom_input_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.bottom_output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def encode_features(self, interaction):
        current_x = torch.flatten(self.current_view(interaction), start_dim=1)
        seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
        history_x = self.att_pool(target_emb, seq_emb, seq_len.view(-1))
        features = torch.cat([current_x, history_x], dim=-1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        return features, domain_ids

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.bottom(x), domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


LOCAL_MODEL_REGISTRY = {
    "din_sharedbottom": DINSharedBottomModel,
}


def resolve_domain_indices(manager: SchemaManager) -> list[tuple[str, int]]:
    domain_setting = manager.get_setting(manager.domain_field)
    if domain_setting is None or not getattr(domain_setting, "vocab", None):
        raise ValueError("domain setting is not fitted yet")
    return sorted(
        ((domain_name, int(domain_idx)) for domain_name, domain_idx in domain_setting.vocab.items()),
        key=lambda item: item[0],
    )


def build_model(manager: SchemaManager, cfg: KuairandSeqConfig) -> tuple[MSRModel, list[tuple[str, int]]]:
    domain_pairs = resolve_domain_indices(manager)
    model_cls = LOCAL_MODEL_REGISTRY[cfg.model]
    model_kwargs = MODEL_KWARGS.get(cfg.model, {}).copy()
    if getattr(cfg, "extras", None):
        model_kwargs.update(cfg.extras)
    model = model_cls.from_manager(manager, num_domains=len(domain_pairs), **model_kwargs)
    return model, domain_pairs


def build_evaluator(manager: SchemaManager, cfg: KuairandSeqConfig, domain_pairs: list[tuple[str, int]]) -> EvaluatorManager:
    evaluator_manager = EvaluatorManager(
        log_path=Path(manager.work_dir) / cfg.log_name,
        title=cfg.experiment_name,
    )
    evaluator_manager.register("overall", Evaluator("auc"))
    for pos, (_, domain_id) in enumerate(domain_pairs):
        evaluator_manager.register(
            f"domain{pos}",
            Evaluator("auc"),
            filter_fn=DomainFilter(manager.domain_field, domain_id),
        )
    return evaluator_manager


def run_training(manager: SchemaManager, cfg: KuairandSeqConfig, train_path: str, valid_path: str):
    model, domain_pairs = build_model(manager, cfg)
    print("domain order:", domain_pairs)
    train_dataset = ParquetStreamDataset(
        train_path,
        manager,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    valid_dataset = ParquetStreamDataset(
        valid_path,
        manager,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    evaluator_manager = build_evaluator(manager, cfg, domain_pairs)
    params = split_params_by_decay(
        model.named_parameters(),
        weight_decay=cfg.weight_decay,
        no_decay_keywords=["embedding"],
    )
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
    trainer = BaseTrainer(
        model,
        optimizer,
        manager,
        TrainerDataLoaders(train=train_dataset, valid=valid_dataset),
        TrainerComponents(
            evaluator_manager=evaluator_manager,
            early_stepper=EarlyStopper(),
        ),
        cfg,
    )
    trainer.run()


if __name__ == "__main__":
    from betterbole.utils.task_chain import auto_queue

    auto_queue()
    manager = build_manager(cfg)
    train_path, valid_path, test_path = prepare_sequence_dataset(manager, cfg)
    if cfg.prepare_only:
        print("prepare_only=True, skip training.")
    else:
        del test_path
        run_training(manager, cfg, train_path, valid_path)
