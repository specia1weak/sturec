from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType
from .utils import NULL_FALLBACK


class BaseCategoricalSetting(EmbSetting):
    """离散单值特征基类：统一管理词表与 OOV 逻辑。"""

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

    def parse_fit_result(self, result_df: pl.DataFrame):
        rows = result_df.get_column(self.field_name).to_list()[0]
        valid_vals = sorted(
            [
                row[self.field_name]
                for row in rows
                if row[self.field_name] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
            ]
        )
        self._build_vocab_indices(valid_vals)

    def get_transform_expr(self) -> pl.Expr:
        replace_kwargs = (
            {"default": pl.lit(self.oov_idx, dtype=pl.UInt32)}
            if self.oov_idx >= 0
            else {}
        )
        return (
            pl.col(self.field_name)
            .cast(pl.Utf8)
            .fill_null(NULL_FALLBACK)
            .replace_strict(self.vocab, **replace_kwargs)
            .cast(pl.UInt32)
            .alias(self.field_name)
        )

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
        return emb_modules[self.field_name](interaction[self.field_name])


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
        return emb_modules[self.field_name](interaction[self.field_name])
