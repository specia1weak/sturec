from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import torch

from betterbole.emb.schema import (
    BaseSequenceSetting,
    EmbSetting,
    EmbType,
    IdSeqEmbSetting,
    SeqDenseSetting,
    SeqGroupEmbSetting,
    SharedVocabSeqSetting,
    SparseSetEmbSetting,
)


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


def build_formatters_from_setting(setting: EmbSetting) -> Dict[str, ColumnFormatter]:
    if isinstance(setting, SeqGroupEmbSetting):
        formatters = {
            seq_col: _build_sequence_formatter(
                max_len=setting.max_len,
                padding_side=getattr(setting, "padding_side", "right"),
                target_setting=target_setting,
            )
            for seq_col, target_setting in setting.target_dict.items()
        }
        formatters[setting.seq_len_field_name] = IntFormatter()
        return formatters

    if isinstance(setting, SeqDenseSetting):
        return {
            setting.field_name: PaddedFloatSequenceFormatter(
                max_len=setting.max_len,
                padding_side=getattr(setting, "padding_side", "right"),
            ),
            setting.seq_len_field_name: IntFormatter(),
        }

    if isinstance(setting, SparseSetEmbSetting):
        return {
            setting.field_name: PaddedIntSequenceFormatter(
                max_len=setting.max_len,
                padding_side=getattr(setting, "padding_side", "right"),
            )
        }

    if isinstance(setting, BaseSequenceSetting):
        return {
            setting.field_name: _build_sequence_formatter(
                max_len=setting.max_len,
                padding_side=getattr(setting, "padding_side", "right"),
                target_setting=_resolve_target_setting(setting),
            ),
            setting.seq_len_field_name: IntFormatter(),
        }

    if setting.emb_type == EmbType.DENSE:
        return {setting.field_name: DenseFormatter()}

    return {setting.field_name: IntFormatter()}


def _resolve_target_setting(setting: BaseSequenceSetting):
    if isinstance(setting, SharedVocabSeqSetting):
        return setting.target_setting
    if isinstance(setting, IdSeqEmbSetting):
        return setting.target_item_setting
    return None


def _build_sequence_formatter(
    max_len: int,
    padding_side: PaddingSide = "right",
    target_setting=None,
) -> ColumnFormatter:
    if isinstance(target_setting, SparseSetEmbSetting):
        return PaddedNestedSequenceFormatter(
            max_seq_len=max_len,
            max_tag_len=target_setting.max_len,
            padding_side=padding_side,
        )
    return PaddedIntSequenceFormatter(max_len=max_len, padding_side=padding_side)
