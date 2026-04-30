from dataclasses import dataclass
from tempfile import TemporaryDirectory

import polars as pl
import torch
from torch.utils.data import DataLoader

from betterbole.core.enum_type import FeatureSource
from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import OmniEmbLayer
from betterbole.emb.schema import (
    SeqDenseSetting,
    SeqGroupConfig,
    SeqGroupEmbSetting,
    SharedVocabSeqSetting,
    SparseEmbSetting,
    SparseSeqEmbSetting,
    SparseSetEmbSetting,
)
from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.utils.sequential import extract_history_sequences


@dataclass
class SmokeConfig(ConfigBase):
    experiment_name: str = "schema-smoke"
    batch_size: int = 2


cfg: SmokeConfig = ParamManager(SmokeConfig).build()


def build_synthetic_lf() -> pl.LazyFrame:
    user_df = pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "age": ["18-24", "25-34", "35-44"],
            "gender": ["M", "F", "M"],
            "occupation": ["student", "engineer", "artist"],
        }
    )
    item_df = pl.DataFrame(
        {
            "movie_id": ["m1", "m2", "m3", "m4"],
            "genres": [
                ["Action", "Sci-Fi"],
                ["Drama"],
                ["Comedy", "Romance"],
                ["Thriller"],
            ],
        }
    )
    inter_df = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
            "movie_id": ["m1", "m2", "m3", "m2", "m4", "m1"],
            "rating": [5.0, 3.0, 4.0, 2.0, 5.0, 4.0],
            "timestamp": [10, 20, 30, 15, 25, 18],
            "search_terms": [
                ["hero", "space"],
                ["sad"],
                ["funny", "date"],
                ["sad", "dark"],
                ["suspense"],
                [],
            ],
            "recent_score_seq": [
                [0.8, 0.9],
                [0.2],
                [0.4, 0.5, 0.6],
                [0.1, 0.3],
                [0.7],
                [],
            ],
            "catalog_movie_seq": [
                ["m1", "m2"],
                ["m2"],
                ["m1", "m3", "m4"],
                ["m2", "m4"],
                ["m4"],
                ["m1"],
            ],
            "catalog_genres_seq": [
                [["Action"], ["Drama"]],
                [["Drama"]],
                [["Action"], ["Comedy", "Romance"], ["Thriller"]],
                [["Drama"], ["Thriller"]],
                [["Thriller"]],
                [["Action", "Sci-Fi"]],
            ],
        }
    )

    whole_lf = (
        inter_df.lazy()
        .join(item_df.lazy(), on="movie_id", how="left")
        .join(user_df.lazy(), on="user_id", how="left")
        .with_columns((pl.col("rating") >= 4).cast(pl.Int8).alias("label"))
    )

    return extract_history_sequences(
        whole_lf,
        max_seq_len=3,
        user_col="user_id",
        time_col="timestamp",
        feature_mapping={
            "movie_id": "movie_id_seq",
            "genres": "genres_seq",
        },
        label_col="label",
        seq_len_col="seq_len",
    )


def build_settings():
    history_group = SeqGroupConfig(
        group_name="history",
        seq_len_field_name="seq_len",
        max_len=3,
        padding_side="right",
    )
    search_group = SeqGroupConfig(
        group_name="search",
        seq_len_field_name="search_len",
        max_len=4,
        padding_side="left",
    )

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, 8, min_freq=1, use_oov=True)
    item_setting = SparseEmbSetting("movie_id", FeatureSource.ITEM_ID, 8, min_freq=1, use_oov=True)
    genres_setting = SparseSetEmbSetting("genres", FeatureSource.ITEM, 6, max_len=3, min_freq=1, use_oov=True)

    return [
        user_setting,
        item_setting,
        SparseEmbSetting("age", FeatureSource.USER, 4, min_freq=1, use_oov=True),
        SparseEmbSetting("gender", FeatureSource.USER, 4, min_freq=1, use_oov=True),
        SparseEmbSetting("occupation", FeatureSource.USER, 4, min_freq=1, use_oov=True),
        genres_setting,
        SharedVocabSeqSetting("movie_id_seq", item_setting, group=history_group),
        SharedVocabSeqSetting("genres_seq", genres_setting, group=history_group),
        SparseSeqEmbSetting("search_terms", search_group, embedding_dim=6, is_string_format=False),
        SeqDenseSetting(
            "recent_score_seq",
            FeatureSource.INTERACTION,
            max_len=4,
            seq_len_field_name="recent_score_len",
            is_string_format=False,
        ),
        SeqGroupEmbSetting(
            "catalog_group",
            "catalog_group_len",
            target_dict={
                "catalog_movie_seq": item_setting,
                "catalog_genres_seq": genres_setting,
            },
            max_len=3,
            is_string_format=False,
        ),
    ]


def main():
    whole_lf = build_synthetic_lf()
    settings = build_settings()

    with TemporaryDirectory() as td:
        manager = SchemaManager(
            settings,
            td,
            time_field="timestamp",
            label_fields="label",
            domain_fields="gender",
        )

        manager.fit(whole_lf, low_memory=False)
        train_lf = manager.transform(whole_lf)
        train_df = train_lf.collect()

        expected_fields = manager.fields()
        missing_fields = [field for field in expected_fields if field not in train_df.columns]
        assert not missing_fields, f"Missing transformed fields: {missing_fields}"

        assert train_df.schema["seq_len"] == pl.UInt32
        assert train_df.schema["search_len"] == pl.UInt32
        assert train_df.schema["recent_score_len"] == pl.UInt32
        assert train_df.schema["catalog_group_len"] == pl.UInt32

        train_path, _, _ = manager.save_as_dataset(train_lf, None, None, output_dir=td, redo=True)

        dataset = ParquetStreamDataset(
            train_path,
            manager,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )
        loader = DataLoader(dataset, batch_size=None, num_workers=0)
        batch = next(iter(loader))

        assert "seq_len" in batch
        assert "search_len" in batch
        assert "recent_score_len" in batch
        assert "catalog_group_len" in batch

        omni = OmniEmbLayer(manager=manager)

        history_seq, history_target, history_len = omni.seq_groups["history"].fetch_all(batch)
        assert history_seq.shape[0] == cfg.batch_size
        assert history_target.shape[0] == cfg.batch_size
        assert torch.equal(history_len, batch["seq_len"])

        named_embs = omni(
            batch,
            split_by="name",
            include_fields=[
                "movie_id_seq",
                "genres_seq",
                "search_terms",
                "recent_score_seq",
                "catalog_group",
            ],
        )
        assert named_embs["movie_id_seq"].shape[0] == cfg.batch_size
        assert named_embs["genres_seq"].shape[0] == cfg.batch_size
        assert named_embs["search_terms"].shape[0] == cfg.batch_size
        assert named_embs["recent_score_seq"].shape[0] == cfg.batch_size
        assert named_embs["catalog_group"].shape[0] == cfg.batch_size

        print("Schema smoke test passed.")
        print("Transformed columns:", train_df.columns)
        print("Batch columns:", batch.columns)
        print("History seq shape:", tuple(history_seq.shape))
        print("History target shape:", tuple(history_target.shape))
        print("Catalog group shape:", tuple(named_embs["catalog_group"].shape))


if __name__ == "__main__":
    main()
