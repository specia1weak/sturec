from typing import Any, Dict, List

import polars as pl
import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource

from .base import EmbType, SeqGroupConfig
from .categorical import BaseCategoricalSetting, SparseEmbSetting
from .utils import clear_seq_expr, explode_expr, map_list_to_indices, mean_pooling


class BaseSequenceSetting(BaseCategoricalSetting):
    """序列特征基类：统一管理清洗、截断和安全类型转换。"""

    def __init__(
        self,
        field_name: str,
        embedding_dim: int,
        source: FeatureSource,
        max_len: int,
        seq_len_field_name: str | None = None,
        is_string_format: bool = False,
        separator: str = ",",
        **kwargs,
    ):
        super().__init__(field_name, embedding_dim, source, **kwargs)
        self.max_len = max_len
        self.seq_len_field_name = seq_len_field_name or f"{field_name}_len"
        self.is_string_format = is_string_format
        self.separator = separator

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["seq_len_field_name"] = self.seq_len_field_name
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.seq_len_field_name = state_dict.get("seq_len_field_name", self.seq_len_field_name)

    def get_clean_expr(self, fill_empty_with_fallback: bool = False) -> pl.Expr:
        return clear_seq_expr(
            field_name=self.field_name,
            is_string_format=self.is_string_format,
            separator=self.separator,
            fill_empty_with_fallback=fill_empty_with_fallback,
        )

    def get_seq_len_expr(self, field_name: str | None = None) -> pl.Expr:
        seq_field = field_name or self.field_name
        raw_len = clear_seq_expr(
            field_name=seq_field,
            is_string_format=self.is_string_format,
            separator=self.separator,
            fill_empty_with_fallback=False,
        ).list.len()
        return self.get_truncated_len_expr(raw_len)

    def get_truncated_len_expr(self, raw_len_expr: pl.Expr) -> pl.Expr:
        return (
            pl.min_horizontal(
                raw_len_expr.cast(pl.UInt32),
                pl.lit(self.max_len, dtype=pl.UInt32),
            )
            .alias(self.seq_len_field_name)
        )


class SparseSetEmbSetting(BaseSequenceSetting):
    emb_type = EmbType.SPARSE_SET

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        embedding_dim: int = 16,
        max_len: int = 5,
        seq_len_field_name: str | None = None,
        is_string_format: bool = False,
        separator: str = ",",
        padding_zero: bool = True,
        min_freq: int = 1,
        use_oov: bool = True,
        agg: str = "sum",
    ):
        super().__init__(
            field_name=field_name,
            embedding_dim=embedding_dim,
            source=source,
            max_len=max_len,
            seq_len_field_name=seq_len_field_name,
            is_string_format=is_string_format,
            separator=separator,
            padding_zero=padding_zero,
            use_oov=use_oov,
            min_freq=min_freq,
        )
        self.agg = agg

    def get_fit_exprs(self) -> List[pl.Expr]:
        exploded = explode_expr(self.field_name, self.is_string_format, self.separator)
        return [exploded.value_counts().implode().alias(self.field_name)]

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
        expr = self.get_clean_expr(fill_empty_with_fallback=self.use_oov).list.tail(self.max_len)
        return map_list_to_indices(expr, self.vocab, self.oov_idx).alias(self.field_name)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_freq"] = self.min_freq
        data["max_len"] = self.max_len
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        data["agg"] = self.agg
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)
        self.max_len = state_dict.get("max_len", 5)
        self.is_string_format = state_dict.get("is_string_format", False)
        self.separator = state_dict.get("separator", ",")
        self.agg = state_dict.get("agg", "sum")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparseSetEmbSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data["embedding_size"],
            max_len=data.get("max_len", 5),
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
        emb = emb_modules[self.field_name](idx_tensor)
        emb = torch.sum(emb, dim=-2)
        if self.agg == "mean":
            emb = mean_pooling(emb, idx_tensor, self.padding_zero)
        return emb


class IdSeqEmbSetting(BaseSequenceSetting):
    emb_type = EmbType.SPARSE_SEQ

    def __init__(
        self,
        field_name: str,
        seq_len_field_name: str,
        target_setting: SparseEmbSetting,
        max_len: int = 50,
        is_string_format: bool = False,
        separator: str = ",",
    ):
        super().__init__(
            field_name=field_name,
            embedding_dim=target_setting.embedding_dim,
            source=target_setting.source,
            max_len=max_len,
            seq_len_field_name=seq_len_field_name,
            is_string_format=is_string_format,
            separator=separator,
            padding_zero=target_setting.padding_zero,
            use_oov=target_setting.use_oov,
        )
        self.target_item_setting = target_setting
        self.is_fitted = True

    @property
    def serialized_type_name(self) -> str:
        return "SHARE_SEQ"

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name, self.emb_type.name}

    @property
    def requires_embedding_module(self) -> bool:
        return False

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
        return

    def get_transform_expr(self) -> List[pl.Expr]:
        expr = self.get_clean_expr(fill_empty_with_fallback=self.use_oov).list.tail(self.max_len)
        mapped_expr = map_list_to_indices(expr, self.vocab, self.oov_idx).alias(self.field_name)
        return [mapped_expr, self.get_seq_len_expr()]

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["max_len"] = self.max_len
        data["target_field_name"] = self.target_item_setting.field_name
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.max_len = state_dict.get("max_len", self.max_len)
        self.is_string_format = state_dict.get("is_string_format", self.is_string_format)
        self.separator = state_dict.get("separator", self.separator)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdSeqEmbSetting":
        raise NotImplementedError("IdSeqEmbSetting 需由 Manager 统一构建依赖关联。")

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return emb_modules[self.target_item_setting.field_name](interaction[self.field_name])


class SeqGroupEmbSetting(BaseSequenceSetting):
    emb_type = EmbType.SPARSE_SEQ

    def __init__(
        self,
        group_name: str,
        seq_len_field_name: str,
        target_dict: Dict[str, BaseCategoricalSetting],
        max_len: int = 50,
        is_string_format: bool = False,
        separator: str = ",",
        combiner: str = "concat",
    ):
        dims = [setting.embedding_dim for setting in target_dict.values()]
        total_dim = sum(dims) if combiner == "concat" else list(target_dict.values())[0].embedding_dim
        super().__init__(
            field_name=group_name,
            embedding_dim=total_dim,
            source=FeatureSource.SEQ_GROUP,
            max_len=max_len,
            seq_len_field_name=seq_len_field_name,
            is_string_format=is_string_format,
            separator=separator,
            padding_zero=True,
            use_oov=False,
        )
        self.target_dict = target_dict
        self.combiner = combiner
        self.is_fitted = True

    def get_fit_exprs(self) -> List[pl.Expr]:
        return []

    def parse_fit_result(self, result_df: pl.DataFrame):
        return

    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def num_embeddings(self) -> int:
        return -1

    @property
    def serialized_type_name(self) -> str:
        return "SEQ_GROUP"

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name, self.emb_type.name}

    @property
    def requires_embedding_module(self) -> bool:
        return False

    def get_transform_expr(self) -> List[pl.Expr]:
        exprs: List[pl.Expr] = []
        for seq_col, target in self.target_dict.items():
            keys = pl.Series(list(target.vocab.keys()), dtype=pl.Utf8)
            vals = pl.Series(list(target.vocab.values()), dtype=pl.UInt32)

            if isinstance(target, SparseSetEmbSetting):
                expr = pl.col(seq_col).fill_null([])
                if self.is_string_format:
                    expr = expr.list.eval(
                        pl.element()
                        .cast(pl.Utf8)
                        .fill_null("")
                        .str.split(self.separator)
                        .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
                    )
                mapped_expr = (
                    expr.list.eval(
                        pl.element().list.eval(
                            pl.element()
                            .cast(pl.Utf8)
                            .replace_strict(
                                old=keys,
                                new=vals,
                                default=pl.lit(target.oov_idx, dtype=pl.UInt32),
                            )
                            .cast(pl.UInt32)
                        ).list.tail(target.max_len)
                    )
                    .list.tail(self.max_len)
                    .alias(seq_col)
                )
            else:
                expr = clear_seq_expr(
                    field_name=seq_col,
                    is_string_format=self.is_string_format,
                    separator=self.separator,
                    fill_empty_with_fallback=target.use_oov,
                )
                mapped_expr = map_list_to_indices(expr.list.tail(self.max_len), target.vocab, target.oov_idx).alias(
                    seq_col
                )
            exprs.append(mapped_expr)
        first_seq_col = next(iter(self.target_dict))
        first_target = self.target_dict[first_seq_col]
        if isinstance(first_target, SparseSetEmbSetting):
            len_expr = self.get_truncated_len_expr(pl.col(first_seq_col).fill_null([]).list.len())
        else:
            len_expr = self.get_seq_len_expr(field_name=first_seq_col)
        exprs.append(len_expr)
        return exprs

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["target_dict"] = {seq_col: target.field_name for seq_col, target in self.target_dict.items()}
        data["max_len"] = self.max_len
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        data["combiner"] = self.combiner
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeqGroupEmbSetting":
        raise NotImplementedError("SeqGroup 涉及对象依赖，需在 Manager 统一反序列化时注入。")

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        group_embs = []
        for seq_col, target in self.target_dict.items():
            mock_interaction = {target.field_name: interaction[seq_col]}
            group_embs.append(target.compute_tensor(mock_interaction, emb_modules))

        if self.combiner == "concat":
            return torch.cat(group_embs, dim=-1)
        if self.combiner == "sum":
            return torch.sum(torch.stack(group_embs, dim=0), dim=0)
        raise ValueError(f"未知 combiner: {self.combiner}")


class SharedVocabSeqSetting(BaseSequenceSetting):
    emb_type = EmbType.SPARSE_SEQ

    def __init__(
        self,
        field_name: str,
        target_setting: BaseCategoricalSetting,
        group: SeqGroupConfig,
        is_string_format: bool = False,
        separator: str = ",",
    ):
        super().__init__(
            field_name=field_name,
            embedding_dim=target_setting.embedding_dim,
            source=FeatureSource.SEQ,
            max_len=group.max_len,
            seq_len_field_name=group.seq_len_field_name,
            is_string_format=is_string_format,
            separator=separator,
            padding_zero=target_setting.padding_zero,
            use_oov=target_setting.use_oov,
        )
        self.target_setting = target_setting
        self.group_name = group.group_name
        self.padding_side = group.padding_side
        self.is_fitted = True

    @property
    def serialized_type_name(self) -> str:
        return "SHARE_SEQ"

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name, self.emb_type.name}

    @property
    def requires_embedding_module(self) -> bool:
        return False

    def get_fit_exprs(self) -> List[pl.Expr]:
        return []

    def parse_fit_result(self, result_df: pl.DataFrame):
        return

    def get_transform_expr(self) -> List[pl.Expr]:
        if isinstance(self.target_setting, SparseSetEmbSetting):
            expr = pl.col(self.field_name).fill_null([])
            if self.is_string_format:
                expr = expr.list.eval(
                    pl.element()
                    .cast(pl.Utf8)
                    .fill_null("")
                    .str.split(self.separator)
                    .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
                )
            keys = pl.Series(list(self.target_setting.vocab.keys()), dtype=pl.Utf8)
            vals = pl.Series(list(self.target_setting.vocab.values()), dtype=pl.UInt32)
            mapped_expr = (
                expr.list.eval(
                    pl.element().list.eval(
                        pl.element()
                        .cast(pl.Utf8)
                        .replace_strict(
                            old=keys,
                            new=vals,
                            default=pl.lit(self.target_setting.oov_idx, dtype=pl.UInt32),
                        )
                        .cast(pl.UInt32)
                    ).list.tail(self.target_setting.max_len)
                )
                .list.tail(self.max_len)
                .alias(self.field_name)
            )
            len_expr = self.get_truncated_len_expr(pl.col(self.field_name).fill_null([]).list.len())
            return [mapped_expr, len_expr]

        expr = self.get_clean_expr(fill_empty_with_fallback=self.target_setting.use_oov).list.tail(self.max_len)
        mapped_expr = map_list_to_indices(expr, self.target_setting.vocab, self.target_setting.oov_idx).alias(
            self.field_name
        )
        return [mapped_expr, self.get_seq_len_expr()]

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["target_field"] = self.target_setting.field_name
        data["group_name"] = self.group_name
        data["max_len"] = self.max_len
        data["padding_side"] = self.padding_side
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedVocabSeqSetting":
        raise NotImplementedError("序列配置对象需在 Python 代码中显式构建依赖。")

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        mock_interaction = {self.target_setting.field_name: interaction[self.field_name]}
        return self.target_setting.compute_tensor(mock_interaction, emb_modules)


class SparseSeqEmbSetting(BaseSequenceSetting):
    emb_type = EmbType.SPARSE_SEQ

    def __init__(
        self,
        field_name: str,
        group: SeqGroupConfig,
        embedding_dim: int = 16,
        is_string_format: bool = False,
        separator: str = ",",
        padding_zero: bool = True,
        min_freq: int = 1,
        use_oov: bool = True,
    ):
        super().__init__(
            field_name=field_name,
            embedding_dim=embedding_dim,
            source=FeatureSource.SEQ,
            max_len=group.max_len,
            seq_len_field_name=group.seq_len_field_name,
            is_string_format=is_string_format,
            separator=separator,
            padding_zero=padding_zero,
            use_oov=use_oov,
            min_freq=min_freq,
        )
        self.group_name = group.group_name
        self.padding_side = group.padding_side

    def get_fit_exprs(self) -> List[pl.Expr]:
        exploded = explode_expr(self.field_name, self.is_string_format, self.separator)
        return [exploded.value_counts().implode().alias(self.field_name)]

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

    def get_transform_expr(self) -> List[pl.Expr]:
        expr = self.get_clean_expr(fill_empty_with_fallback=self.use_oov).list.tail(self.max_len)
        mapped_expr = map_list_to_indices(expr, self.vocab, self.oov_idx).alias(self.field_name)

        len_expr = self.get_seq_len_expr()
        return [mapped_expr, len_expr]

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["group_name"] = self.group_name
        data["min_freq"] = self.min_freq
        data["max_len"] = self.max_len
        data["padding_side"] = self.padding_side
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.group_name = state_dict.get("group_name")
        self.min_freq = state_dict.get("min_freq", 1)
        self.max_len = state_dict.get("max_len", 50)
        self.padding_side = state_dict.get("padding_side", "right")
        self.is_string_format = state_dict.get("is_string_format", False)
        self.separator = state_dict.get("separator", ",")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparseSeqEmbSetting":
        raise NotImplementedError("序列配置对象需在 Python 代码中显式构建依赖。")

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return emb_modules[self.field_name](interaction[self.field_name])


class SeqDenseSetting(BaseSequenceSetting):
    emb_type = EmbType.DENSE_SEQ

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        max_len: int,
        seq_len_field_name: str | None = None,
        min_val: float = None,
        max_val: float = None,
        is_string_format: bool = False,
        separator: str = ",",
    ):
        super().__init__(
            field_name=field_name,
            embedding_dim=1,
            source=source,
            max_len=max_len,
            seq_len_field_name=seq_len_field_name,
            is_string_format=is_string_format,
            separator=separator,
            padding_zero=False,
            use_oov=False,
        )
        self.min_val = min_val
        self.max_val = max_val
        self.is_fitted = min_val is not None and max_val is not None

    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def num_embeddings(self) -> int:
        return -1

    def _float_seq_expr(self) -> pl.Expr:
        if self.is_string_format:
            return (
                pl.col(self.field_name)
                .cast(pl.Utf8)
                .fill_null("")
                .str.split(self.separator)
                .list.eval(
                    pl.when(pl.element().str.strip_chars() != "")
                    .then(pl.element().str.strip_chars().cast(pl.Float32, strict=False))
                    .otherwise(None)
                )
                .list.drop_nulls()
            )
        return (
            pl.col(self.field_name)
            .cast(pl.List(pl.Float32))
            .fill_null([])
            .list.eval(pl.element().cast(pl.Float32, strict=False).filter(pl.element().is_not_null()))
        )

    def get_fit_exprs(self) -> List[pl.Expr]:
        exploded = self._float_seq_expr().explode()
        return [
            exploded.min().alias(f"{self.field_name}_min"),
            exploded.max().alias(f"{self.field_name}_max"),
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

    def get_transform_expr(self) -> List[pl.Expr]:
        expr = self._float_seq_expr().list.tail(self.max_len)
        if self.is_fitted:
            range_val = self.max_val - self.min_val
            expr = expr.list.eval(((pl.element() - self.min_val) / range_val).fill_null(0.0).cast(pl.Float32))
        mapped_expr = expr.alias(self.field_name)
        len_expr = self.get_truncated_len_expr(self._float_seq_expr().list.len())
        return [mapped_expr, len_expr]

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_val"] = self.min_val
        data["max_val"] = self.max_val
        data["max_len"] = self.max_len
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_val = state_dict.get("min_val")
        self.max_val = state_dict.get("max_val")
        self.max_len = state_dict.get("max_len", self.max_len)
        self.is_string_format = state_dict.get("is_string_format", self.is_string_format)
        self.separator = state_dict.get("separator", self.separator)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeqDenseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            max_len=data.get("max_len", 50),
            seq_len_field_name=data.get("seq_len_field_name"),
            min_val=data.get("min_val"),
            max_val=data.get("max_val"),
            is_string_format=data.get("is_string_format", False),
            separator=data.get("separator", ","),
        )
        obj.load_state(data)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return interaction[self.field_name].to(torch.float32)
