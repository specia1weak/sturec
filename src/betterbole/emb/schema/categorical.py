from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType
from .utils import NULL_FALLBACK, map_list_to_indices, mean_pooling


class BaseCategoricalSetting(EmbSetting):
    """离散特征基类：统一管理词表与 OOV 逻辑。"""

    def __init__(
        self,
        field_name: str,
        embedding_dim: int,
        source: FeatureSource,
        padding_zero: bool = True,
        use_oov: bool = True,
        min_freq: int = 1,
    ):
        super().__init__(field_name, embedding_dim, source, padding_zero, use_oov)
        self.min_freq = min_freq
        self.vocab = {}
        self.oov_idx = -1

    def _build_vocab_indices(self, valid_vals: List[str]):
        start_idx = 1 if self.padding_zero else 0
        self.vocab = {str(val): idx + start_idx for idx, val in enumerate(valid_vals)}
        if self.use_oov:
            self.oov_idx = len(self.vocab) + start_idx
        else:
            self.oov_idx = 0 if self.padding_zero else -1
        self.is_fitted = True


class SparseEmbSetting(BaseCategoricalSetting):
    emb_type = EmbType.SPARSE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        embedding_dim: int = 16,
        padding_zero: bool = True,
        min_freq: int = 1,
        use_oov: bool = True,
    ):
        super().__init__(field_name, embedding_dim, source, padding_zero, use_oov, min_freq=min_freq)

    def get_fit_exprs(self) -> List[pl.Expr]:
        return [
            pl.col(self.field_name)
            .cast(pl.Utf8)
            .drop_nulls()
            .value_counts()
            .implode()
            .alias(self.field_name)
        ]

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        exploded_raw = seq_expr.explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(alias)]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        rows = result_df.get_column(alias).to_list()[0]
        valid_vals = sorted(
            [
                row[alias]
                for row in rows
                if row[alias] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
            ]
        )
        self._build_vocab_indices(valid_vals)

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        replace_kwargs = {"default": pl.lit(self.oov_idx, dtype=pl.UInt32)} if self.oov_idx >= 0 else {}
        return (
            expr.cast(pl.Utf8)
            .fill_null(NULL_FALLBACK)
            .replace_strict(self.vocab, **replace_kwargs)
            .cast(pl.UInt32)
        )

    def get_transform_expr(self) -> pl.Expr:
        return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)

    def get_formatters(self):
        from betterbole.data.padding import IntFormatter

        return {self.field_name: IntFormatter()}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_freq"] = self.min_freq
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparseEmbSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data["embedding_size"],
            padding_zero=data.get("padding_zero", True),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", True),
        )
        obj.load_state(data)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return emb_modules[self.embedding_field_name](interaction[self.field_name])


class QuantileEmbSetting(EmbSetting):
    emb_type = EmbType.QUANTILE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        bucket_count: int = 10,
        embedding_dim: int = 16,
        boundaries: Optional[List[float]] = None,
    ):
        super().__init__(field_name, embedding_dim, source, padding_zero=True, use_oov=False)
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
        return [
            pl.col(self.field_name).drop_nulls().cast(pl.Float64).quantile(q).alias(f"{self.field_name}_q_{idx}")
            for idx, q in enumerate(q_list)
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        boundaries = []
        for idx in range(self.bucket_count - 1):
            value = result_df.get_column(f"{self.field_name}_q_{idx}")[0]
            if value is not None and not np.isnan(value):
                boundaries.append(value)
        self.boundaries = sorted(set(boundaries))
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        if not self.is_fitted or not self.boundaries:
            return pl.lit(0, dtype=pl.UInt32).alias(self.field_name)

        bucket_count = len(self.boundaries) + 1
        labels = [str(idx) for idx in range(1, bucket_count + 1)]
        return (
            pl.col(self.field_name)
            .cut(breaks=self.boundaries, labels=labels)
            .cast(pl.String)
            .cast(pl.UInt32)
            .fill_null(0)
            .alias(self.field_name)
        )

    def get_formatters(self):
        from betterbole.data.padding import IntFormatter

        return {self.field_name: IntFormatter()}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["bucket_count"] = self.bucket_count
        data["boundaries"] = self.boundaries
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.bucket_count = state_dict.get("bucket_count", self.bucket_count)
        self.boundaries = state_dict.get("boundaries", [])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantileEmbSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            bucket_count=data.get("bucket_count", 10),
            embedding_dim=data["embedding_size"],
            boundaries=data.get("boundaries", []),
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return emb_modules[self.embedding_field_name](interaction[self.field_name])


class MultiSparseSetting(BaseCategoricalSetting):
    emb_type = EmbType.MULTI_SPARSE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        embedding_dim: int = 16,
        max_tag_len: int = 5,
        is_string_format: bool = False,
        separator: str = ",",
        padding_zero: bool = True,
        min_freq: int = 1,
        use_oov: bool = True,
        agg: str = "sum",
    ):
        super().__init__(field_name, embedding_dim, source, padding_zero, use_oov, min_freq=min_freq)
        self.max_tag_len = max_tag_len
        self.is_string_format = is_string_format
        self.separator = separator
        self.agg = agg

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name, EmbType.SPARSE_SET.name}

    @property
    def fill_empty_sequence_with_fallback(self) -> bool:
        return False

    def _clean_list_expr(self, expr: pl.Expr, fill_empty_with_fallback: bool = False) -> pl.Expr:
        if self.is_string_format:
            cleaned = (
                expr.cast(pl.Utf8)
                .fill_null("")
                .str.split(self.separator)
                .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
            )
        else:
            cleaned = (
                expr.cast(pl.List(pl.Utf8))
                .fill_null([])
                .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
            )

        if not fill_empty_with_fallback:
            return cleaned

        return (
            pl.when(cleaned.list.len() > 0)
            .then(cleaned)
            .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
        )

    def _truncate_tag_expr(self, expr: pl.Expr) -> pl.Expr:
        return expr.list.head(self.max_tag_len)

    def get_fit_exprs(self) -> List[pl.Expr]:
        exploded_raw = self._truncate_tag_expr(self._clean_list_expr(pl.col(self.field_name))).explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(self.field_name)]

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        nested_expr = seq_expr.list.eval(
            self._truncate_tag_expr(self._clean_list_expr(pl.element(), fill_empty_with_fallback=False))
        )
        exploded_raw = nested_expr.explode().explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(alias)]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        rows = result_df.get_column(alias).to_list()[0]
        valid_vals = sorted(
            [
                row[alias]
                for row in rows
                if row[alias] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
            ]
        )
        self._build_vocab_indices(valid_vals)

    def get_transform_expr(self) -> pl.Expr:
        expr = self._truncate_tag_expr(
            self._clean_list_expr(pl.col(self.field_name), fill_empty_with_fallback=False)
        )
        return map_list_to_indices(expr, self.vocab, self.oov_idx).alias(self.field_name)

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        cleaned = self._truncate_tag_expr(
            self._clean_list_expr(expr, fill_empty_with_fallback=False)
        )
        return map_list_to_indices(cleaned, self.vocab, self.oov_idx)

    def get_formatters(self):
        from betterbole.data.padding import PaddedIntSequenceFormatter

        return {self.field_name: PaddedIntSequenceFormatter(max_len=self.max_tag_len)}

    def get_sequence_formatters(self, max_len: int, padding_side: str):
        from betterbole.data.padding import PaddedNestedSequenceFormatter

        return {
            self.field_name: PaddedNestedSequenceFormatter(
                max_seq_len=max_len,
                max_tag_len=self.max_tag_len,
                padding_side=padding_side,
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_freq"] = self.min_freq
        data["max_tag_len"] = self.max_tag_len
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        data["agg"] = self.agg
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)
        self.max_tag_len = state_dict.get("max_tag_len", state_dict.get("max_len", self.max_tag_len))
        self.is_string_format = state_dict.get("is_string_format", False)
        self.separator = state_dict.get("separator", ",")
        self.agg = state_dict.get("agg", "sum")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiSparseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data["embedding_size"],
            max_tag_len=data.get("max_tag_len", data.get("max_len", 5)),
            is_string_format=data.get("is_string_format", False),
            separator=data.get("separator", ","),
            padding_zero=data.get("padding_zero", True),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", True),
            agg=data.get("agg", "sum"),
        )
        obj.load_state(data)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        idx_tensor = interaction[self.field_name]
        emb = emb_modules[self.embedding_field_name](idx_tensor)
        emb = torch.sum(emb, dim=-2)
        if self.agg == "mean":
            emb = mean_pooling(emb, idx_tensor, self.padding_zero)
        return emb
