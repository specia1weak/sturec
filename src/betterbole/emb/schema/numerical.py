from typing import Any, Dict, List

import polars as pl
import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType


class BaseNumericalSetting(EmbSetting):
    """连续数值特征基类：显式声明无词表。"""

    def __init__(self, field_name: str, source: FeatureSource, embedding_dim: int = 1):
        super().__init__(
            field_name=field_name,
            embedding_dim=embedding_dim,
            source=source,
            padding_zero=False,
            use_oov=False,
        )
        self.is_fitted = False

    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def num_embeddings(self) -> int:
        return -1

    @property
    def requires_embedding_module(self) -> bool:
        return False


class MinMaxDenseSetting(BaseNumericalSetting):
    emb_type = EmbType.DENSE

    def __init__(self, field_name: str, source: FeatureSource, min_val: float = None, max_val: float = None):
        super().__init__(field_name=field_name, source=source, embedding_dim=1)
        self.min_val = min_val
        self.max_val = max_val
        self.is_fitted = min_val is not None and max_val is not None

    def get_fit_exprs(self) -> List[pl.Expr]:
        return [
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().min().alias(f"{self.field_name}_min"),
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().max().alias(f"{self.field_name}_max"),
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
        return (
            ((pl.col(self.field_name) - self.min_val) / range_val)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(self.field_name)
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_val"] = self.min_val
        data["max_val"] = self.max_val
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_val = state_dict.get("min_val")
        self.max_val = state_dict.get("max_val")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinMaxDenseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            min_val=data.get("min_val"),
            max_val=data.get("max_val"),
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return interaction[self.field_name].unsqueeze(-1)


class VectorDenseSetting(BaseNumericalSetting):
    emb_type = EmbType.DENSE

    def __init__(self, field_name: str, source: FeatureSource, vector_dim: int, zero_fill: bool = True):
        super().__init__(field_name=field_name, source=source, embedding_dim=vector_dim)
        self.vector_dim = vector_dim
        self.zero_fill = zero_fill
        self.is_fitted = True

    def get_fit_exprs(self) -> List[pl.Expr]:
        return []

    def parse_fit_result(self, result_df: pl.DataFrame):
        return

    def get_transform_expr(self) -> pl.Expr:
        zero_vector = [0.0] * self.vector_dim
        expr = pl.col(self.field_name).cast(pl.List(pl.Float32))
        if self.zero_fill:
            expr = expr.fill_null(zero_vector)
        return (
            pl.when(expr.list.len() == self.vector_dim)
            .then(expr)
            .otherwise(pl.lit(zero_vector, dtype=pl.List(pl.Float32)))
            .alias(self.field_name)
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["vector_dim"] = self.vector_dim
        data["zero_fill"] = self.zero_fill
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.vector_dim = state_dict.get("vector_dim", self.vector_dim)
        self.zero_fill = state_dict.get("zero_fill", self.zero_fill)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDenseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            vector_dim=data["vector_dim"],
            zero_fill=data.get("zero_fill", True),
        )
        obj.is_fitted = data.get("is_fitted", True)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return interaction[self.field_name].to(torch.float32)
