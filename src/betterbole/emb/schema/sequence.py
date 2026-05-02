from typing import Any, Dict, List, Optional

import polars as pl
import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType, SeqGroupConfig
from .utils import NULL_FALLBACK


def _dedupe_names(names: List[Optional[str]]) -> List[str]:
    return list(dict.fromkeys(name for name in names if name))


def _clean_outer_sequence_expr(
    expr: pl.Expr,
    is_string_format: bool,
    separator: str,
    fill_empty_with_fallback: bool = False,
) -> pl.Expr:
    if is_string_format:
        cleaned = (
            expr.cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        )
    elif fill_empty_with_fallback:
        cleaned = (
            expr.cast(pl.List(pl.Utf8))
            .fill_null([])
            .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
        )
    else:
        cleaned = expr.fill_null([]).list.eval(pl.element().filter(pl.element().is_not_null()))

    if not fill_empty_with_fallback:
        return cleaned

    return (
        pl.when(cleaned.list.len() > 0)
        .then(cleaned)
        .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
    )


def _clean_time_sequence_expr(expr: pl.Expr, is_string_format: bool, separator: str) -> pl.Expr:
    if is_string_format:
        return (
            expr.cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(
                pl.when(pl.element().str.strip_chars() != "")
                .then(pl.element().str.strip_chars().cast(pl.Int64, strict=False))
                .otherwise(None)
            )
            .list.drop_nulls()
        )
    return (
        expr.cast(pl.List(pl.Int64))
        .fill_null([])
        .list.eval(pl.element().cast(pl.Int64, strict=False).filter(pl.element().is_not_null()))
    )


class SequenceSetting(EmbSetting):
    emb_type = EmbType.SEQUENCE

    def __init__(
        self,
        field_name: str,
        element_setting: EmbSetting,
        group: SeqGroupConfig,
        truncate_mode: str = "tail",
        is_string_format: bool = False,
        separator: str = ",",
    ):
        super().__init__(
            field_name=field_name,
            embedding_dim=element_setting.embedding_dim,
            source=FeatureSource.SEQ,
            padding_zero=element_setting.padding_zero,
            use_oov=element_setting.use_oov,
        )
        self.element_setting = element_setting
        self.group = group
        self.max_len = group.max_len
        self.group_name = group.group_name
        self.seq_len_field_name = group.seq_len_field_name
        self.padding_side = group.padding_side
        self.time_field_name = group.time_field_name
        self.truncate_mode = truncate_mode
        self.is_string_format = is_string_format
        self.separator = separator
        self.is_fitted = element_setting.is_fitted

    @property
    def compatible_type_names(self) -> set[str]:
        legacy = {self.serialized_type_name, EmbType.SPARSE_SEQ.name, EmbType.DENSE_SEQ.name, "SHARE_SEQ"}
        return legacy

    @property
    def is_sequence_setting(self) -> bool:
        return True

    @property
    def embedding_field_name(self) -> str:
        return self.element_setting.embedding_field_name

    @property
    def vocab_size(self) -> int:
        return self.element_setting.vocab_size

    @property
    def num_embeddings(self) -> int:
        return self.element_setting.num_embeddings

    @property
    def requires_embedding_module(self) -> bool:
        return self.element_setting.requires_embedding_module

    def get_output_field_names(self) -> List[str]:
        return _dedupe_names([self.field_name, self.seq_len_field_name, self.time_field_name])

    def get_raw_field_names(self) -> List[str]:
        return _dedupe_names([self.field_name, self.time_field_name])

    def get_clean_expr(self, fill_empty_with_fallback: bool = False, field_name: Optional[str] = None) -> pl.Expr:
        seq_field = field_name or self.field_name
        return _clean_outer_sequence_expr(
            pl.col(seq_field),
            is_string_format=self.is_string_format,
            separator=self.separator,
            fill_empty_with_fallback=fill_empty_with_fallback,
        )

    def truncate_seq_expr(self, expr: pl.Expr) -> pl.Expr:
        if self.truncate_mode == "head":
            return expr.list.head(self.max_len)
        if self.truncate_mode == "tail":
            return expr.list.tail(self.max_len)
        raise ValueError(f"Unknown truncate_mode: {self.truncate_mode}")

    def get_truncated_len_expr(self, raw_len_expr: pl.Expr) -> pl.Expr:
        return (
            pl.min_horizontal(
                raw_len_expr.cast(pl.UInt32),
                pl.lit(self.max_len, dtype=pl.UInt32),
            )
            .alias(self.seq_len_field_name)
        )

    def get_seq_len_expr(self, field_name: Optional[str] = None) -> pl.Expr:
        raw_len = self.get_clean_expr(fill_empty_with_fallback=False, field_name=field_name).list.len()
        return self.get_truncated_len_expr(raw_len)

    def get_fit_seq_expr(self) -> pl.Expr:
        return self.truncate_seq_expr(
            self.get_clean_expr(fill_empty_with_fallback=False)
        )

    def get_fit_exprs(self) -> List[pl.Expr]:
        if self.element_setting.is_fitted:
            self.is_fitted = True
            return []
        return self.element_setting.get_element_fit_exprs(self.get_fit_seq_expr(), alias=self.field_name)

    def parse_fit_result(self, result_df: pl.DataFrame):
        if self.element_setting.is_fitted:
            self.is_fitted = True
            return
        self.element_setting.parse_element_fit_result(result_df, alias=self.field_name)
        self.is_fitted = self.element_setting.is_fitted

    def get_time_transform_expr(self) -> Optional[pl.Expr]:
        if not self.time_field_name:
            return None
        expr = _clean_time_sequence_expr(
            pl.col(self.time_field_name),
            is_string_format=self.is_string_format,
            separator=self.separator,
        )
        return self.truncate_seq_expr(expr).alias(self.time_field_name)

    def get_transform_expr(self) -> List[pl.Expr]:
        expr = self.truncate_seq_expr(
            self.get_clean_expr(fill_empty_with_fallback=False)
        )
        mapped_expr = expr.list.eval(self.element_setting.get_element_transform_expr()).alias(self.field_name)
        exprs: List[pl.Expr] = [mapped_expr, self.get_seq_len_expr()]
        time_expr = self.get_time_transform_expr()
        if time_expr is not None:
            exprs.append(time_expr)
        return exprs

    def get_formatters(self) -> Dict[str, "ColumnFormatter"]:
        from betterbole.data.padding import IntFormatter, PaddedIntSequenceFormatter

        element_formatter = next(
            iter(
                self.element_setting.get_sequence_formatters(
                    max_len=self.max_len,
                    padding_side=self.padding_side,
                ).values()
            )
        )
        formatters = {
            self.field_name: element_formatter,
            self.seq_len_field_name: IntFormatter(),
        }
        if self.time_field_name:
            formatters[self.time_field_name] = PaddedIntSequenceFormatter(
                max_len=self.max_len,
                padding_side=self.padding_side,
            )
        return formatters

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["group_name"] = self.group_name
        data["seq_len_field_name"] = self.seq_len_field_name
        data["max_len"] = self.max_len
        data["padding_side"] = self.padding_side
        data["time_field_name"] = self.time_field_name
        data["truncate_mode"] = self.truncate_mode
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        data["element_setting_field_name"] = self.element_setting.field_name
        data["element_setting_type"] = self.element_setting.serialized_type_name
        data["element_setting_state"] = self.element_setting.to_dict()
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        self.group_name = state_dict.get("group_name", self.group_name)
        self.seq_len_field_name = state_dict.get("seq_len_field_name", self.seq_len_field_name)
        self.max_len = state_dict.get("max_len", self.max_len)
        self.padding_side = state_dict.get("padding_side", self.padding_side)
        self.time_field_name = state_dict.get("time_field_name", self.time_field_name)
        self.truncate_mode = state_dict.get("truncate_mode", self.truncate_mode)
        self.is_string_format = state_dict.get("is_string_format", self.is_string_format)
        self.separator = state_dict.get("separator", self.separator)
        element_state = state_dict.get("element_setting_state")
        if element_state:
            self.element_setting.load_state(element_state)
        self.is_fitted = self.element_setting.is_fitted

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SequenceSetting":
        raise NotImplementedError("SequenceSetting 需在 Python 代码中显式注入 element_setting 依赖。")

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        mock_interaction = {
            self.field_name: interaction[self.field_name],
            self.element_setting.field_name: interaction[self.field_name],
        }
        return self.element_setting.compute_tensor(mock_interaction, emb_modules)
