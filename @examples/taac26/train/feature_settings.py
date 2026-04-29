from __future__ import annotations

from typing import List

from betterbole.core.enum_type import FeatureSource
from betterbole.emb.schema import EmbSetting, SparseEmbSetting, SparseSetEmbSetting

from config import TAACConfig


def build_sparse_settings(cfg: TAACConfig) -> List[EmbSetting]:
    settings: List[EmbSetting] = []

    # The official baseline does not use raw top-level ids directly inside the
    # model; it first encodes them into synthetic user/item int features during
    # preprocessing. In this local flat-parquet path we approximate that by
    # embedding the top-level ids directly when enabled.
    if cfg.include_top_level_ids:
        settings.append(
            SparseEmbSetting(
                "user_id",
                FeatureSource.USER_ID,
                cfg.id_emb,
                min_freq=cfg.min_freq,
                use_oov=True,
            )
        )
        settings.append(
            SparseEmbSetting(
                "item_id",
                FeatureSource.ITEM_ID,
                cfg.id_emb,
                min_freq=cfg.min_freq,
                use_oov=True,
            )
        )

    for col in cfg.official_user_sparse_scalar_fields:
        settings.append(
            SparseEmbSetting(
                col,
                FeatureSource.USER,
                embedding_dim=cfg.sparse_emb,
                min_freq=cfg.min_freq,
            )
        )

    for col in cfg.official_item_sparse_scalar_fields:
        settings.append(
            SparseEmbSetting(
                col,
                FeatureSource.ITEM,
                embedding_dim=cfg.sparse_emb,
                min_freq=cfg.min_freq,
            )
        )

    # for col in cfg.official_user_sparse_seq_fields:
    #     settings.append(
    #         SparseSetEmbSetting(
    #             col,
    #             FeatureSource.USER,
    #             embedding_dim=cfg.sparse_emb,
    #             min_freq=cfg.min_freq,
    #             max_len=cfg.profile_set_max_len,
    #             agg=cfg.sparse_set_agg,
    #         )
    #     )
    #
    # for col in cfg.official_item_sparse_seq_fields:
    #     settings.append(
    #         SparseSetEmbSetting(
    #             col,
    #             FeatureSource.ITEM,
    #             embedding_dim=cfg.sparse_emb,
    #             min_freq=cfg.min_freq,
    #             max_len=cfg.profile_set_max_len,
    #             agg=cfg.sparse_set_agg,
    #         )
    #     )
    #
    # for col in cfg.official_sequence_sparse_seq_fields:
    #     settings.append(
    #         SparseSetEmbSetting(
    #             col,
    #             FeatureSource.INTERACTION,
    #             embedding_dim=cfg.sparse_emb,
    #             min_freq=cfg.min_freq,
    #             max_len=cfg.seq_max_len,
    #             agg=cfg.sparse_set_agg,
    #         )
    #     )
    #
    return settings
