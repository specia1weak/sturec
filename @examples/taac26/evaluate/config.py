from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from betterbole.experiment.param import ConfigBase, ParamManager


SplitStrategy = Literal["loo", "time", "sequential_ratio", "random_ratio"]
SparseSetAgg = Literal["sum", "mean"]
EvalUidConflictStrategy = Literal["strict", "overwrite", "agg_mean", "agg_max", "agg_last"]


def _prefixed_range(prefix: str, start: int, end: int) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(start, end + 1)]


# Official-code-backed grouping notes:
# - In `reference-projects/kddsample/preprocess_raw_to_kddsample.py`, every raw
#   feature with an integer branch goes into schema["user_int"] /
#   schema["item_int"] / schema["seq"][...]["features"].
# - In `reference-projects/kddsample/dataset.py`, those groups are materialized
#   into `user_int_feats`, `item_int_feats`, and domain sequence tensors.
# - In `reference-projects/kddsample/model.py`, all three routes are embedded by
#   `nn.Embedding`, while `user_dense` is projected by `nn.Linear`.
# So for the flat official parquet, every `*_int_feats_*` and `domain_*_seq_*`
# column should be treated as sparse/discrete under the baseline logic.

OFFICIAL_USER_INT_SCALAR_FIELDS = [
    "user_int_feats_1",
    "user_int_feats_3",
    "user_int_feats_4",
    *_prefixed_range("user_int_feats", 48, 59),
    "user_int_feats_82",
    "user_int_feats_86",
    *_prefixed_range("user_int_feats", 92, 109),
]

OFFICIAL_USER_INT_LIST_FIELDS = [
    "user_int_feats_15",
    "user_int_feats_60",
    *_prefixed_range("user_int_feats", 62, 66),
    "user_int_feats_80",
    *_prefixed_range("user_int_feats", 89, 91),
]

OFFICIAL_USER_DENSE_FIELDS = [
    *_prefixed_range("user_dense_feats", 61, 66),
    "user_dense_feats_87",
    *_prefixed_range("user_dense_feats", 89, 91),
]

OFFICIAL_ITEM_INT_SCALAR_FIELDS = [
    *_prefixed_range("item_int_feats", 5, 10),
    *_prefixed_range("item_int_feats", 12, 13),
    "item_int_feats_16",
    "item_int_feats_81",
    *_prefixed_range("item_int_feats", 83, 85),
]

OFFICIAL_ITEM_INT_LIST_FIELDS = ["item_int_feats_11"]

OFFICIAL_DOMAIN_A_SEQ_FIELDS = _prefixed_range("domain_a_seq", 38, 46)
OFFICIAL_DOMAIN_B_SEQ_FIELDS = [
    *_prefixed_range("domain_b_seq", 67, 79),
    "domain_b_seq_88",
]
OFFICIAL_DOMAIN_C_SEQ_FIELDS = [
    *_prefixed_range("domain_c_seq", 27, 37),
    "domain_c_seq_47",
]
OFFICIAL_DOMAIN_D_SEQ_FIELDS = _prefixed_range("domain_d_seq", 17, 26)

OFFICIAL_ALL_SEQUENCE_FIELDS = [
    *OFFICIAL_DOMAIN_A_SEQ_FIELDS,
    *OFFICIAL_DOMAIN_B_SEQ_FIELDS,
    *OFFICIAL_DOMAIN_C_SEQ_FIELDS,
    *OFFICIAL_DOMAIN_D_SEQ_FIELDS,
]

OFFICIAL_ALL_SPARSE_SCALAR_FIELDS = [
    "user_id",
    "item_id",
    *OFFICIAL_USER_INT_SCALAR_FIELDS,
    *OFFICIAL_ITEM_INT_SCALAR_FIELDS,
]

OFFICIAL_ALL_SPARSE_SET_FIELDS = [
    *OFFICIAL_USER_INT_LIST_FIELDS,
    *OFFICIAL_ITEM_INT_LIST_FIELDS,
    *OFFICIAL_ALL_SEQUENCE_FIELDS,
]


@dataclass
class TAACConfig(ConfigBase):
    dataset_name: str = "taac26"
    experiment_name: str = "taac26-local-sim"
    seed: int = 2026
    device: str = "cuda"
    max_epochs: int = 1

    batch_size: int = 4096
    valid_batch_size: int = 8192
    shuffle_buffer_size: int = 2_000_000

    id_emb: int = 32
    sparse_emb: int = 16
    min_freq: int = 10
    sparse_set_agg: SparseSetAgg = "mean"
    profile_set_max_len: int = 16
    seq_max_len: int = 64
    include_top_level_ids: bool = True
    include_official_dense_float_lists: bool = False

    hidden_dims: list = field(default_factory=lambda: [256, 128])

    split_strategy: SplitStrategy = "sequential_ratio"
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    label_shift: int = 1
    eval_uid_conflict_strategy: EvalUidConflictStrategy = "overwrite"

    @property
    def official_user_sparse_scalar_fields(self) -> list[str]:
        return list(OFFICIAL_USER_INT_SCALAR_FIELDS)

    @property
    def official_user_sparse_seq_fields(self) -> list[str]:
        return list(OFFICIAL_USER_INT_LIST_FIELDS)

    @property
    def official_item_sparse_scalar_fields(self) -> list[str]:
        return list(OFFICIAL_ITEM_INT_SCALAR_FIELDS)

    @property
    def official_item_sparse_seq_fields(self) -> list[str]:
        return list(OFFICIAL_ITEM_INT_LIST_FIELDS)

    @property
    def official_sequence_sparse_seq_fields(self) -> list[str]:
        return list(OFFICIAL_ALL_SEQUENCE_FIELDS)

    @property
    def official_dense_float_list_fields(self) -> list[str]:
        return list(OFFICIAL_USER_DENSE_FIELDS)


def build_cfg() -> TAACConfig:
    pm = ParamManager(TAACConfig)
    return pm.build()
