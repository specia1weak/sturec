from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Literal, Optional

import numpy as np
import torch

from betterbole.core.interaction import Interaction

if TYPE_CHECKING:
    from betterbole.emb import SchemaManager


PaddingSide = Literal["left", "right"]


def _is_missing_sequence(value) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _pad_list(
    sequences,
    max_len: int,
    padding_side: PaddingSide = "right",
    pad_val=0,
    dtype=np.int64,
) -> np.ndarray:
    padded = np.full((len(sequences), max_len), pad_val, dtype=dtype)
    for i, seq in enumerate(sequences):
        if _is_missing_sequence(seq) or len(seq) == 0:
            continue

        seq_list = list(seq)
        valid_len = min(len(seq_list), max_len)
        trunc_seq = seq_list[-valid_len:]
        if padding_side == "right":
            padded[i, :valid_len] = trunc_seq
        elif padding_side == "left":
            padded[i, max_len - valid_len:] = trunc_seq
        else:
            raise ValueError(f"Unknown padding_side: {padding_side}")

    return padded


def _pad_nested_list(
    sequences,
    max_seq_len: int,
    max_tag_len: int,
    padding_side: PaddingSide = "right",
    pad_val: int = 0,
) -> np.ndarray:
    padded = np.full((len(sequences), max_seq_len, max_tag_len), pad_val, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if _is_missing_sequence(seq) or len(seq) == 0:
            continue

        seq_list = list(seq)
        valid_seq_len = min(len(seq_list), max_seq_len)
        trunc_seq = seq_list[-valid_seq_len:]
        start_idx = 0 if padding_side == "right" else max_seq_len - valid_seq_len

        for j in range(valid_seq_len):
            tags = trunc_seq[j]
            if _is_missing_sequence(tags) or len(tags) == 0:
                continue

            tags_list = list(tags)
            valid_tag_len = min(len(tags_list), max_tag_len)
            trunc_tags = tags_list[:valid_tag_len]
            padded[i, start_idx + j, :valid_tag_len] = trunc_tags

    return padded


class ColumnFormatter(ABC):
    @abstractmethod
    def format(self, data) -> torch.Tensor:
        pass


@dataclass(frozen=True)
class DenseFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)


@dataclass(frozen=True)
class VectorDenseFormatter(ColumnFormatter):
    dim: int
    zero_fill: bool = True

    def format(self, data) -> torch.Tensor:
        arr = np.zeros((len(data), self.dim), dtype=np.float32)
        for i, row in enumerate(data):
            if _is_missing_sequence(row) or len(row) == 0:
                if not self.zero_fill:
                    raise ValueError(
                        f"Expected non-empty vector dim={self.dim}, but got an empty value at row {i}."
                    )
                continue

            row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
            use_len = min(row_arr.shape[0], self.dim)
            arr[i, :use_len] = row_arr[:use_len]

        return torch.tensor(arr, dtype=torch.float32)


@dataclass(frozen=True)
class IntFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        if data.dtype == np.uint32:
            data = data.astype(np.int64)
        return torch.tensor(data, dtype=torch.long)


@dataclass(frozen=True)
class FallbackFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        if data.dtype in (np.float32, np.float64):
            return torch.tensor(data, dtype=torch.float32)
        if data.dtype == np.uint32:
            return torch.tensor(data.astype(np.int64), dtype=torch.long)
        return torch.tensor(data)


@dataclass(frozen=True)
class PaddedIntSequenceFormatter(ColumnFormatter):
    max_len: int
    padding_side: PaddingSide = "right"
    pad_val: int = 0

    def format(self, data) -> torch.Tensor:
        arr = _pad_list(
            data,
            max_len=self.max_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
            dtype=np.int64,
        )
        return torch.tensor(arr, dtype=torch.long)


@dataclass(frozen=True)
class PaddedFloatSequenceFormatter(ColumnFormatter):
    max_len: int
    padding_side: PaddingSide = "right"
    pad_val: float = 0.0

    def format(self, data) -> torch.Tensor:
        arr = _pad_list(
            data,
            max_len=self.max_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
            dtype=np.float32,
        )
        return torch.tensor(arr, dtype=torch.float32)


@dataclass(frozen=True)
class PaddedNestedSequenceFormatter(ColumnFormatter):
    max_seq_len: int
    max_tag_len: int
    padding_side: PaddingSide = "right"
    pad_val: int = 0

    def format(self, data) -> torch.Tensor:
        arr = _pad_nested_list(
            data,
            max_seq_len=self.max_seq_len,
            max_tag_len=self.max_tag_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
        )
        return torch.tensor(arr, dtype=torch.long)


DEFAULT_FALLBACK_FORMATTER = FallbackFormatter()


def _dedupe_names(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(name for name in names if name))


def _resolve_raw_read_col_names(
    manager: "SchemaManager",
    extra_col_names: Iterable[str],
    extra_col_formatters: Dict[str, ColumnFormatter],
) -> list[str]:
    raw_col_names: list[str] = []
    for setting in manager.settings:
        raw_col_names.extend(setting.get_raw_field_names())

    raw_col_names.extend(manager.label_fields)
    raw_col_names.extend(manager.domain_fields)
    if manager.time_field:
        raw_col_names.append(manager.time_field)

    raw_col_names.extend(extra_col_names)
    raw_col_names.extend(extra_col_formatters.keys())
    return _dedupe_names(raw_col_names)


@dataclass(frozen=True)
class FeatureContext:
    manager: "SchemaManager"
    read_col_names: tuple[str, ...]
    output_col_names: tuple[str, ...]
    extra_col_formatters: Dict[str, ColumnFormatter] = field(default_factory=dict)

    @classmethod
    def from_manager(
        cls,
        manager: "SchemaManager",
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ) -> "FeatureContext":
        formatter_map = dict(extra_col_formatters or {})
        extra_names = list(extra_col_names or [])
        output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
        return cls(
            manager=manager,
            read_col_names=tuple(output_cols),
            output_col_names=tuple(output_cols),
            extra_col_formatters=formatter_map,
        )

    @classmethod
    def from_raw_manager(
        cls,
        manager: "SchemaManager",
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ) -> "FeatureContext":
        formatter_map = dict(extra_col_formatters or {})
        extra_names = list(extra_col_names or [])
        output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
        read_cols = _resolve_raw_read_col_names(manager, extra_names, formatter_map)
        return cls(
            manager=manager,
            read_col_names=tuple(read_cols),
            output_col_names=tuple(output_cols),
            extra_col_formatters=formatter_map,
        )


@dataclass
class TensorFormatter:
    context: FeatureContext
    fallback_formatter: ColumnFormatter = DEFAULT_FALLBACK_FORMATTER
    col_formatters: Dict[str, ColumnFormatter] = field(init=False)

    def __post_init__(self):
        self.col_formatters = self._compile_formatters()

    def _compile_formatters(self) -> Dict[str, ColumnFormatter]:
        formatters: Dict[str, ColumnFormatter] = {}
        for setting in self.context.manager.settings:
            formatters.update(setting.get_formatters())

        for ctx_col in (
            *self.context.manager.label_fields,
            *self.context.manager.domain_fields,
            self.context.manager.time_field,
        ):
            if ctx_col and ctx_col not in formatters:
                formatters[ctx_col] = self.fallback_formatter

        formatters.update(self.context.extra_col_formatters)
        return formatters

    def format_tensors(self, batch_dict) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for col, data in batch_dict.items():
            formatter = self.col_formatters.get(col, self.fallback_formatter)
            try:
                tensor_dict[col] = formatter.format(data)
            except Exception as exc:
                raise RuntimeError(
                    f"列 [{col}] 在转换为 Tensor 时发生错误，匹配到的 formatter 为: {formatter}。错误信息: {exc}"
                ) from exc
        return tensor_dict

    def format(self, batch_dict) -> Interaction:
        return Interaction(self.format_tensors(batch_dict))
