from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import polars as pl
import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource

if TYPE_CHECKING:
    from betterbole.data.padding import ColumnFormatter


class EmbType(Enum):
    UNKNOWN = "none"
    SPARSE = "sparse"
    MULTI_SPARSE = "multi_sparse"
    QUANTILE = "quantile"
    SPARSE_SEQ = "sparse_seq"
    SPARSE_SET = "sparse_set"
    DENSE = "dense"
    VECTOR_DENSE = "vector_dense"
    DENSE_SEQ = "dense_seq"
    SEQUENCE = "sequence"


@dataclass
class SeqGroupConfig:
    group_name: str
    seq_len_field_name: str
    max_len: int
    padding_side: Literal["left", "right"] = "right"
    time_field_name: Optional[str] = None


class EmbSetting(ABC):
    emb_type = EmbType.UNKNOWN

    def __init__(
        self,
        field_name: str,
        embedding_dim: int,
        source: FeatureSource = FeatureSource.UNKNOWN,
        padding_zero: bool = True,
        use_oov: bool = True,
    ):
        self.field_name = field_name
        self.embedding_dim = embedding_dim
        self.source = source
        self.is_fitted = False

        self.padding_zero = padding_zero
        self.use_oov = use_oov
        self.vocab: Dict[str, int] = {}
        self.oov_idx: int = -1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return self.vocab_size + int(self.padding_zero) + int(self.use_oov)

    @property
    def serialized_type_name(self) -> str:
        return self.emb_type.name

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name}

    @property
    def requires_embedding_module(self) -> bool:
        return self.num_embeddings > 0

    @property
    def embedding_field_name(self) -> str:
        return self.field_name

    @property
    def is_sequence_setting(self) -> bool:
        return False

    def get_output_field_names(self) -> List[str]:
        return [self.field_name]

    def get_raw_field_names(self) -> List[str]:
        return [self.field_name]

    @abstractmethod
    def get_fit_exprs(self) -> List[pl.Expr]:
        pass

    @abstractmethod
    def parse_fit_result(self, result_df: pl.DataFrame):
        pass

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        return self.parse_fit_result(result_df)

    @abstractmethod
    def get_transform_expr(self):
        pass

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element fitting.")

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element transform.")

    @abstractmethod
    def get_formatters(self) -> Dict[str, "ColumnFormatter"]:
        pass

    def get_sequence_formatters(
        self,
        max_len: int,
        padding_side: Literal["left", "right"],
    ) -> Dict[str, "ColumnFormatter"]:
        from betterbole.data.padding import (
            DenseFormatter,
            IntFormatter,
            PaddedFloatSequenceFormatter,
            PaddedIntSequenceFormatter,
        )

        base_formatters = self.get_formatters()
        fmt = base_formatters[self.field_name]
        if isinstance(fmt, DenseFormatter):
            return {self.field_name: PaddedFloatSequenceFormatter(max_len=max_len, padding_side=padding_side)}
        if isinstance(fmt, IntFormatter):
            return {self.field_name: PaddedIntSequenceFormatter(max_len=max_len, padding_side=padding_side)}
        raise NotImplementedError(
            f"Formatter {fmt.__class__.__name__} cannot be promoted to a sequence automatically."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.serialized_type_name,
            "field_name": self.field_name,
            "embedding_size": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "vocab_size": self.vocab_size,
            "feature_source": self.source.name,
            "is_fitted": self.is_fitted,
            "padding_zero": self.padding_zero,
            "use_oov": self.use_oov,
            "vocab": self.vocab,
            "oov_idx": self.oov_idx,
        }

    def load_state(self, state_dict: Dict[str, Any]):
        self.is_fitted = state_dict.get("is_fitted", True)
        self.vocab = state_dict.get("vocab", {})
        self.use_oov = state_dict.get("use_oov", self.use_oov)
        self.padding_zero = state_dict.get("padding_zero", self.padding_zero)
        self.oov_idx = state_dict.get("oov_idx", self.oov_idx)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbSetting":
        pass

    @abstractmethod
    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        pass
