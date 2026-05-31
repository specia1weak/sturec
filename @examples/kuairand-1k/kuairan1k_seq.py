from dataclasses import dataclass
from typing import Tuple

import polars as pl

from betterbole.core.enum_type import FeatureSource
from betterbole.datasets.kuairand import KuaiRandDataset
from betterbole.emb import SchemaManager
from betterbole.emb.schema import SeqGroupConfig, SequenceSetting, SparseEmbSetting, MultiSparseSetting
from betterbole.experiment import WORKSPACE, change_root_workdir
from betterbole.utils.sequential import extract_history_sequences

change_root_workdir()


@dataclass
class KuairandSeqConfig:
    dataset_name: str = "kuairand-rand-seq-v1"
    seed: int = 2026
    id_emb: int = 32
    side_emb: int = 32
    seq_max_len: int = 20
    top_k_domains: int = 5
    min_freq: int = 10
    log_name: str = "seq_prepare.log"


cfg = KuairandSeqConfig()


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


def build_joined_frame() -> pl.LazyFrame:
    user_lf = pl.scan_csv(KuaiRandDataset.USER_FEATURES)
    item_lf = pl.scan_csv(KuaiRandDataset.VIDEO_FEATURES)
    inter_lf = pl.scan_csv(
        [
            KuaiRandDataset.STD_LOG_FORMER_DATA_P1,
            KuaiRandDataset.STD_LOG_FORMER_DATA_P2,
            KuaiRandDataset.RAND_LOG_FORMER_DATA,
        ]
    )
    whole_lf = (
        inter_lf.join(item_lf, on="video_id", how="left")
        .join(user_lf, on="user_id", how="left")
    )

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


def prepare_sequence_dataset() -> Tuple[str, str, str]:
    work_dir = WORKSPACE / cfg.dataset_name
    settings = build_settings(cfg)
    manager = SchemaManager(
        settings,
        work_dir,
        time_field="time_ms",
        label_fields="is_click",
        domain_fields="tab",
    )

    whole_lf = build_joined_frame()
    whole_lf = manager.make_checkpoint(
        whole_lf,
        file_name="kuairand_seq_joined.parquet",
        redo=False,
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

    train_lf = manager.transform(train_raw).select(manager.fields())
    valid_lf = manager.transform(valid_raw).select(manager.fields())
    test_lf = manager.transform(test_raw).select(manager.fields())

    print("schema fields:", manager.fields())
    print("train sample:")
    print(
        train_lf.select(
            [
                "user_id",
                "video_id_seq",
                "tab_seq",
                "author_id_seq",
                "music_id_seq",
                "video_type_seq",
                "music_type_seq",
                "visible_status_seq",
                "tag_seq",
                "time_ms_seq",
                "seq_len",
                "is_click",
                "tab",
            ]
        )
        .head(3)
        .collect()
    )

    train_path, valid_path, test_path = manager.save_as_dataset(
        train_lf,
        valid_lf,
        test_lf,
        redo=False,
    )
    print("sequence dataset ready:", train_path, valid_path, test_path)
    return train_path, valid_path, test_path


if __name__ == "__main__":
    prepare_sequence_dataset()
