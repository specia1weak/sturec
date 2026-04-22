from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional

import numpy as np
import polars as pl

from betterbole.core.enum_type import FeatureSource


class EmbType(Enum):
    UNKNOWN = "none"
    SPARSE = "sparse"
    ABS_RANGE = "abs_range"
    QUANTILE = "quantile"
    SPARSE_SEQ = "sparse_seq"
    SPARSE_SET = "sparse_set"
    DENSE = "dense"


def explode_expr(field_name, is_string_format=True, separator=","):
    expr = pl.col(field_name).drop_nulls()
    if is_string_format:
        exploded = expr.str.split(separator).explode().str.strip_chars()
        valid_mask = (exploded != "") & exploded.is_not_null()
    else:
        exploded = expr.explode().cast(pl.Utf8)
        valid_mask = exploded.is_not_null() & (exploded != "null")
    return exploded.filter(valid_mask)


def clear_seq_expr(field_name, is_string_format, separator):
    expr = pl.col(field_name)
    if is_string_format:
        expr = expr.fill_null("").str.split(separator)
        clean_expr = pl.element().str.strip_chars()
        filter_expr = pl.element() != ""
    else:
        expr = expr.fill_null([])
        clean_expr = pl.element().cast(pl.Utf8)
        filter_expr = pl.element().is_not_null()
    return expr.list.eval(clean_expr.filter(filter_expr))



# ==========================================
# 1. 规则层 (Rule Layer) - 彻底声明式
# ==========================================
class EmbSetting(ABC):
    emb_type = EmbType.UNKNOWN

    def __init__(self, field_name: str, embedding_size: int, source: FeatureSource = FeatureSource.UNKNOWN,
                 padding_zero=True, use_oov: bool = True):
        self.field_name = field_name
        self.embedding_size = embedding_size
        self.source = source
        self.is_fitted = False


        self.padding_zero = padding_zero
        self.use_oov = use_oov
        self.vocab: Dict[str, int] = {} # 只包括有效词表的映射，不包括oov和pad
        self.oov_idx: int = -1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return self.vocab_size + int(self.padding_zero) + int(self.use_oov)

    def _build_vocab_indices(self, valid_vals: List[str]):
        start_idx = 1 if self.padding_zero else 0
        self.vocab = {str(val): idx + start_idx for idx, val in enumerate(valid_vals)}
        if self.use_oov:
            self.oov_idx = len(self.vocab) + start_idx
        else:
            self.oov_idx = 0 if self.padding_zero else -1
        self.is_fitted = True

    @abstractmethod
    def get_fit_exprs(self) -> List[pl.Expr]:
        pass

    @abstractmethod
    def parse_fit_result(self, result_df: pl.DataFrame):
        pass

    @abstractmethod
    def get_transform_expr(self) -> pl.Expr:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.emb_type.name,
            "field_name": self.field_name,
            "embedding_size": self.embedding_size,
            "num_embeddings": self.num_embeddings,
            "vocab_size": self.vocab_size,
            "feature_source": self.source.name,
            "is_fitted": self.is_fitted,
            "padding_zero": self.padding_zero,
            "use_oov": self.use_oov,
            "vocab": self.vocab,
            "oov_idx": self.oov_idx
        }

    def load_state(self, state_dict: Dict[str, Any]):
        self.is_fitted = state_dict.get("is_fitted", True)
        self.vocab = state_dict.get("vocab", {})
        self.use_oov = state_dict.get("use_oov", self.use_oov)
        self.padding_zero = state_dict.get("padding_zero", self.padding_zero)
        self.oov_idx = state_dict.get("oov_idx", self.oov_idx)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbSetting':
        pass


class SparseEmbSetting(EmbSetting):
    emb_type = EmbType.SPARSE

    def __init__(self, field_name: str, source: FeatureSource, embedding_size: int = 16,
                 padding_zero: bool = True, min_freq: int = 1, use_oov: bool = True):
        super().__init__(field_name, embedding_size, source, padding_zero, use_oov)
        self.min_freq = min_freq

    def get_fit_exprs(self):
        return [
            pl.col(self.field_name).cast(pl.Utf8).drop_nulls()
            .value_counts()
            .implode()
            .alias(self.field_name)
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        rows = result_df.get_column(self.field_name).to_list()[0]
        col_name = self.field_name

        # 只负责提取符合频率要求的词
        valid_vals = sorted([
            r[col_name] for r in rows
            if r[col_name] is not None and r.get("count", r.get("counts", 0)) >= self.min_freq
        ])

        # 核心映射逻辑交由基类处理
        self._build_vocab_indices(valid_vals)

    def get_transform_expr(self):
        replace_kwargs = (
            {"default": pl.lit(self.oov_idx, dtype=pl.UInt32)}
            if self.oov_idx >= 0
            else {}
        )
        return (
            pl.col(self.field_name)
            .fill_null("NULL_FALLBACK")
            .cast(pl.Utf8)
            .replace_strict(self.vocab, **replace_kwargs)
            .cast(pl.UInt32)
            .alias(self.field_name)
        )

    def to_dict(self):
        d = super().to_dict()
        d["min_freq"] = self.min_freq
        return d

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_size=data["embedding_size"],
            padding_zero=data.get("padding_zero", True),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", True)
        )
        obj.load_state(data)
        return obj


class SparseSetEmbSetting(EmbSetting):
    emb_type = EmbType.SPARSE_SET

    def __init__(self, field_name: str, source: FeatureSource, embedding_size: int = 16,
                 max_len: int = 5, is_string_format: bool = False, separator: str = ",",
                 padding_zero: bool = True, min_freq: int = 1, use_oov: bool = True, agg="sum"):
        super().__init__(field_name, embedding_size, source, padding_zero, use_oov)
        self.max_len = max_len
        self.is_string_format = is_string_format
        self.separator = separator
        self.min_freq = min_freq
        self.agg = agg

    def get_fit_exprs(self) -> List[pl.Expr]:
        exploded = explode_expr(self.field_name, self.is_string_format, self.separator)
        final_expr = (
            exploded.value_counts()
            .implode()
            .alias(self.field_name)
        )
        return [final_expr]

    def parse_fit_result(self, result_df: pl.DataFrame):
        rows = result_df.get_column(self.field_name).to_list()[0]
        col_name = self.field_name

        valid_vals = sorted([
            r[col_name] for r in rows
            if r[col_name] is not None and r.get("count", r.get("counts", 0)) >= self.min_freq
        ])

        self._build_vocab_indices(valid_vals)

    def get_transform_expr(self) -> pl.Expr:
        expr = clear_seq_expr(self.field_name, self.is_string_format, self.separator)
        keys = pl.Series(list(self.vocab.keys()), dtype=pl.Utf8)
        vals = pl.Series(list(self.vocab.values()), dtype=pl.UInt32)

        mapped_expr = (
            expr.list.eval(
                pl.element()
                .fill_null("NULL_FALLBACK")
                .cast(pl.Utf8)
                .replace_strict(old=keys, new=vals, default=pl.lit(self.oov_idx, dtype=pl.UInt32))
                .cast(pl.UInt32)
            )
            .list.tail(self.max_len)
            .alias(self.field_name)
        )
        return mapped_expr

    def to_dict(self):
        d = super().to_dict()
        d["min_freq"] = self.min_freq
        d["max_len"] = self.max_len
        d["is_string_format"] = self.is_string_format
        d["separator"] = self.separator
        d["agg"] = self.agg
        return d

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)
        self.max_len = state_dict.get("max_len", 5)

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_size=data["embedding_size"],
            max_len=data.get("max_len", 5),
            is_string_format=data.get("is_string_format", False),
            separator=data.get("separator", ","),
            padding_zero=data.get("padding_zero", True),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", True),
            agg=data.get("agg", "sum")
        )
        obj.load_state(data)
        return obj


class QuantileEmbSetting(EmbSetting):
    emb_type = EmbType.QUANTILE

    def __init__(self, field_name: str, source: FeatureSource, bucket_count: int = 10,
                 embedding_size: int = 16, boundaries: Optional[List[float]] = None):
        # Quantile离散化不需要常规的字符串OOV机制
        super().__init__(field_name, embedding_size, source, padding_zero=True, use_oov=False)
        self.bucket_count = bucket_count
        self.boundaries = boundaries if boundaries is not None else []
        if self.boundaries:
            self.is_fitted = True

    @property
    def vocab_size(self) -> int:
        return len(self.boundaries) + 1 if self.boundaries else 0

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return self.vocab_size + 1

    def get_fit_exprs(self) -> List[pl.Expr]:
        q_list = np.linspace(0, 1, self.bucket_count + 1)[1:-1]
        exprs = []
        for i, q in enumerate(q_list):
            expr = pl.col(self.field_name).drop_nulls().cast(pl.Float64).quantile(q).alias(f"{self.field_name}_q_{i}")
            exprs.append(expr)
        return exprs

    def parse_fit_result(self, result_df: pl.DataFrame):
        bounds = []
        for i in range(self.bucket_count - 1):
            val = result_df.get_column(f"{self.field_name}_q_{i}")[0]
            if val is not None and not np.isnan(val):
                bounds.append(val)
        self.boundaries = sorted(list(set(bounds)))
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        if not self.is_fitted or not self.boundaries:
            return pl.lit(0).alias(self.field_name)
        num_buckets = len(self.boundaries) + 1
        labels = [str(i) for i in range(1, num_buckets + 1)]
        expr = (
            pl.col(self.field_name)
            .cut(breaks=self.boundaries, labels=labels)
            .cast(pl.String)
            .cast(pl.UInt32)
            .fill_null(0)
            .alias(self.field_name)
        )
        return expr

    def to_dict(self):
        d = super().to_dict()
        d["bucket_count"] = self.bucket_count
        d["boundaries"] = self.boundaries
        return d

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.bucket_count = state_dict.get("bucket_count", self.bucket_count)
        self.boundaries = state_dict.get("boundaries", [])

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            bucket_count=data.get("bucket_count", 10),
            embedding_size=data["embedding_size"],
            boundaries=data.get("boundaries", [])
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj


class IdSeqEmbSetting(EmbSetting):
    emb_type = EmbType.SPARSE_SEQ

    def __init__(self, field_name: str, seq_len_field_name: str, target_setting: SparseEmbSetting, max_len: int = 50,
                 is_string_format: bool = False, separator: str = ","):
        # 共享目标词表的配置
        super().__init__(
            field_name=field_name,
            embedding_size=target_setting.embedding_size,
            source=target_setting.source,
            padding_zero=target_setting.padding_zero,
            use_oov=target_setting.use_oov
        )
        self.seq_len_field_name = seq_len_field_name
        self.target_item_setting = target_setting
        self.max_len = max_len
        self.is_string_format = is_string_format
        self.separator = separator
        self.is_fitted = True

    # 动态透传 target_setting 的词汇属性
    @property
    def vocab(self):
        return self.target_item_setting.vocab
    @vocab.setter
    def vocab(self, value):
        return

    @property
    def vocab_size(self) -> int:
        return self.target_item_setting.vocab_size

    @property
    def num_embeddings(self) -> int:
        return self.target_item_setting.num_embeddings

    @property
    def oov_idx(self) -> int:
        return self.target_item_setting.oov_idx

    @oov_idx.setter
    def oov_idx(self, value):
        return

    def get_fit_exprs(self) -> List[pl.Expr]:
        return []

    def parse_fit_result(self, result_df: pl.DataFrame):
        pass

    def get_transform_expr(self) -> pl.Expr:
        expr = clear_seq_expr(self.field_name, self.is_string_format, self.separator)
        keys = pl.Series(list(self.vocab.keys()), dtype=pl.Utf8)
        vals = pl.Series(list(self.vocab.values()), dtype=pl.UInt32)

        mapped_expr = (
            expr.list.eval(
                pl.element()
                .fill_null("NULL_FALLBACK")
                .cast(pl.Utf8)
                .replace_strict(old=keys, new=vals, default=pl.lit(self.oov_idx, dtype=pl.UInt32))
                .cast(pl.UInt32)
            )
            .alias(self.field_name)
        )
        return mapped_expr

    def to_dict(self):
        d = super().to_dict()
        d["max_len"] = self.max_len
        d["target_field_name"] = self.target_item_setting.field_name
        d["is_string_format"] = self.is_string_format
        d["separator"] = self.separator
        d["seq_len_field_name"] = self.seq_len_field_name
        return d

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.max_len = state_dict.get("max_len", self.max_len)

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError("IdSeqEmbSetting 需由 Manager 统一构建依赖关联。")


class MinMaxDenseSetting(EmbSetting):
    emb_type = EmbType.DENSE

    def __init__(self, field_name: str, source: FeatureSource, min_val: float = None, max_val: float = None):
        # 连续值特征不需要任何词表逻辑
        super().__init__(field_name, 1, source, padding_zero=False, use_oov=False)
        self.min_val = min_val
        self.max_val = max_val
        if self.min_val is not None and self.max_val is not None:
            self.is_fitted = True
        else:
            self.is_fitted = False

    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def num_embeddings(self) -> int:
        # 重写属性：Dense 不涉及 Embedding lookup table，直接返回 -1
        return -1

    def get_fit_exprs(self) -> List[pl.Expr]:
        return [
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().min().alias(f"{self.field_name}_min"),
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().max().alias(f"{self.field_name}_max")
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.min_val = result_df.get_column(f"{self.field_name}_min").to_list()[0]
        self.max_val = result_df.get_column(f"{self.field_name}_max").to_list()[0]

        if self.min_val is None or self.max_val is None:
            self.min_val = 0.0
            self.max_val = 1.0
        elif self.min_val == self.max_val:
            self.max_val = self.min_val + 1e-6
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        range_val = self.max_val - self.min_val
        expr = (pl.col(self.field_name) - self.min_val) / range_val
        return expr.fill_null(0.0).cast(pl.Float32).alias(self.field_name)

    def to_dict(self):
        d = super().to_dict()
        d["min_val"] = self.min_val
        d["max_val"] = self.max_val
        return d

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_val = state_dict.get("min_val")
        self.max_val = state_dict.get("max_val")

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            min_val=data.get("min_val"),
            max_val=data.get("max_val")
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj