from typing import Any, Dict, List, Optional

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
    emb_type = EmbType.VECTOR_DENSE

    def __init__(self, field_name: str, source: 'FeatureSource', embedding_dim: Optional[int] = None,
                 zero_fill: bool = True):
        super().__init__(field_name=field_name, source=source, embedding_dim=embedding_dim or 0)
        self.zero_fill = zero_fill
        self.is_fitted = embedding_dim is not None

    def get_fit_exprs(self) -> List[pl.Expr]:
        if self.is_fitted:
            return []

        len_expr = pl.col(self.field_name).drop_nulls().list.len()
        return [
            len_expr.value_counts().implode().alias(f"{self.field_name}_len_counts"),
            pl.col(self.field_name).is_null().sum().alias(f"{self.field_name}_null_count"),
            len_expr.eq(0).sum().alias(f"{self.field_name}_empty_count"),
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        if self.is_fitted:
            return

        len_counts_key = f"{self.field_name}_len_counts"
        len_rows = result_df.get_column(len_counts_key).to_list()[0]
        null_count = int(result_df.get_column(f"{self.field_name}_null_count")[0] or 0)
        empty_count = int(result_df.get_column(f"{self.field_name}_empty_count")[0] or 0)

        length_distribution: Dict[int, int] = {}
        for row in len_rows or []:
            count = row.get("count", row.get("counts", 0))
            length = None
            for key, value in row.items():
                if key not in ("count", "counts"):
                    length = value
                    break
            if length is None:
                continue
            length_distribution[int(length)] = int(count)

        non_empty_distribution = {
            int(length): int(count)
            for length, count in length_distribution.items()
            if int(length) > 0 and int(count) > 0
        }

        if not non_empty_distribution:
            raise ValueError(
                f"Feature '{self.field_name}' has no non-empty vectors in training data. "
                f"null_rows={null_count}, empty_rows={empty_count}. "
                "Please set `embedding_dim` explicitly if this feature should be kept."
            )

        if len(non_empty_distribution) != 1:
            dist_lines = [
                f"  {length} -> {count} rows"
                for length, count in sorted(non_empty_distribution.items())
            ]
            distribution_text = "\n".join(dist_lines)
            raise ValueError(
                f"Feature '{self.field_name}' has inconsistent vector lengths on training data.\n"
                f"Observed non-empty length distribution:\n{distribution_text}\n"
                f"null_rows={null_count}, empty_rows={empty_count}\n"
                "Please set `embedding_dim` explicitly for this feature."
            )

        self.embedding_dim = next(iter(non_empty_distribution))
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        if not self.is_fitted or not self.embedding_dim:
            raise RuntimeError(f"Feature '{self.field_name}' is not fitted yet. embedding_dim is unknown.")

        # Keep the raw float list column. Fixed-width padding/truncation is
        # handled in the dataset formatter so checkpoints preserve the original
        # vector information instead of silently zeroing mismatched rows.
        return pl.col(self.field_name).cast(pl.List(pl.Float32)).alias(self.field_name)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["embedding_dim"] = self.embedding_dim
        data["zero_fill"] = self.zero_fill
        return data

    @property
    def compatible_type_names(self) -> set[str]:
        # Backward compatibility: older checkpoints stored vector dense as DENSE.
        return {self.serialized_type_name, EmbType.DENSE.name}

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.embedding_dim = state_dict.get("embedding_dim", self.embedding_dim)
        self.zero_fill = state_dict.get("zero_fill", self.zero_fill)
        self.is_fitted = self.embedding_dim is not None and self.embedding_dim > 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDenseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data.get("embedding_dim", data.get("embedding_size")),
            zero_fill=data.get("zero_fill", True),
        )
        obj.is_fitted = data.get("is_fitted", obj.embedding_dim is not None)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return interaction[self.field_name].to(torch.float32)
