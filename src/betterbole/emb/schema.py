import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Union

import numpy as np
import polars as pl
from typing import Literal

from src.betterbole.data.split import SPLIT_STRATEGIES, SplitContext
from src.betterbole.enum_type import FeatureSource

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

    def __init__(self, field_name: str, embedding_size: int, source: FeatureSource=FeatureSource.UNKNOWN, padding_zero=True):
        self.field_name = field_name
        self.embedding_size = embedding_size
        self.source = source
        self.is_fitted = False
        self.padding_zero = padding_zero

    @property
    @abstractmethod
    def num_embeddings(self) -> int:
        pass

    @abstractmethod
    def get_fit_exprs(self) -> List[pl.Expr]:
        """【核心变更】不直接扫描数据，而是返回获取统计量（如Unique或Quantile）的表达式"""
        pass

    @abstractmethod
    def parse_fit_result(self, result_df: pl.DataFrame):
        """【核心变更】接收一次性扫描算出的统一结果，更新自身状态"""
        pass

    @abstractmethod
    def get_transform_expr(self) -> pl.Expr:
        """【核心变更】返回修改数据的表达式计算图节点"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.emb_type.name,
            "field_name": self.field_name,
            "embedding_size": self.embedding_size,
            "num_embeddings": self.num_embeddings,
            "feature_source": self.source.name,
            "is_fitted": self.is_fitted
        }

    @abstractmethod
    def load_state(self, state_dict: Dict[str, Any]):
        """将 JSON 字典中的状态注入到当前实例中"""
        # self.embedding_size = state_dict.get("embedding_size", self.embedding_size) # 不推荐通过读取的方式写定embsize，因此注释
        self.is_fitted = state_dict.get("is_fitted", True)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbSetting':
        pass


class SparseEmbSetting(EmbSetting):
    emb_type = EmbType.SPARSE

    def __init__(self, field_name: str, source: FeatureSource, embedding_size: int = 16, num_embeddings: int = -1, padding_zero=True):
        super().__init__(field_name, embedding_size, source, padding_zero)
        self._num_embeddings = num_embeddings
        self.vocab: Dict[str, int] = {}
        if num_embeddings > 0:
            self.is_fitted = True

    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings

    def get_fit_exprs(self) -> List[pl.Expr]:
        # Polars 魔法：去重 -> 剔除空值 -> 打包成一个 List 返回给单行 DataFrame
        return [
            pl.col(self.field_name).cast(pl.Utf8).drop_nulls().unique()
            .implode().alias(self.field_name)
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        # 提取聚合后的列表
        unique_vals = result_df.get_column(self.field_name).to_list()[0]
        unique_vals = [v for v in unique_vals if v is not None]
        unique_vals = sorted(unique_vals)

        # 统一转为字符串作为字典 Key，规避类型坑。预留 0 为 OOV/Padding
        if self.padding_zero:
            self.vocab = {str(val): idx + 1 for idx, val in enumerate(unique_vals)}
            self._num_embeddings = len(self.vocab) + 1
        else:
            self.vocab = {str(val): idx + 0 for idx, val in enumerate(unique_vals)}
            self._num_embeddings = len(self.vocab) + 0
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        return pl.col(self.field_name) \
            .cast(pl.Utf8) \
            .replace_strict(self.vocab, default=pl.lit(0, dtype=pl.UInt32)) \
            .cast(pl.UInt32) \
            .alias(self.field_name)

    def to_dict(self):
        d = super().to_dict()
        d["vocab"] = self.vocab
        return d

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.vocab = state_dict.get("vocab", {})
        self._num_embeddings = state_dict.get("num_embeddings", len(self.vocab) + 1)

    @classmethod
    def from_dict(cls, data):
        obj = cls(data["field_name"], FeatureSource[data["feature_source"]], data["embedding_size"],
                  data["num_embeddings"])
        obj.vocab = data.get("vocab", {})
        obj.is_fitted = data.get("is_fitted", False)
        return obj


class QuantileEmbSetting(EmbSetting):
    emb_type = EmbType.QUANTILE

    def __init__(self, field_name: str, source: FeatureSource, bucket_count: int = 10, embedding_size: int = 16,
                 boundaries: Optional[List[float]] = None):
        super().__init__(field_name, embedding_size, source)
        self.bucket_count = bucket_count
        self.boundaries = boundaries if boundaries is not None else []
        if self.boundaries:
            self.is_fitted = True

    @property
    def num_embeddings(self) -> int:
        return len(self.boundaries) + 2

    def get_fit_exprs(self) -> List[pl.Expr]:
        # 比如分为 10 桶，产生 9 个切分点
        q_list = np.linspace(0, 1, self.bucket_count + 1)[1:-1]
        exprs = []
        for i, q in enumerate(q_list):
            # 将每个分位数的计算定义为独立的列
            expr = pl.col(self.field_name).drop_nulls().cast(pl.Float64).quantile(q).alias(f"{self.field_name}_q_{i}")
            exprs.append(expr)
        return exprs

    def parse_fit_result(self, result_df: pl.DataFrame):
        bounds = []
        for i in range(self.bucket_count - 1):
            val = result_df.get_column(f"{self.field_name}_q_{i}")[0]
            if val is not None and not np.isnan(val):
                bounds.append(val)

        # 去重并排序，生成严谨的分界线
        self.boundaries = sorted(list(set(bounds)))
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        if not self.boundaries:
            # 数据全空的情况，全填 1
            return pl.lit(1, dtype=pl.UInt32).alias(self.field_name)

        # 强转类型 -> 空值填0 -> 按边界切分 -> 取出分箱标签 -> 转整型
        labels = [str(i + 1) for i in range(len(self.boundaries) + 1)]
        return pl.col(self.field_name) \
            .cast(pl.Float64) \
            .fill_null(0.0) \
            .cut(breaks=self.boundaries, labels=labels, left_closed=False) \
            .cast(pl.UInt32) \
            .alias(self.field_name)

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
        obj = cls(data["field_name"], FeatureSource[data["feature_source"]], data.get("bucket_count", 10),
                  data["embedding_size"], data.get("boundaries", []))
        obj.is_fitted = data.get("is_fitted", False)
        return obj

class SparseSetEmbSetting(EmbSetting):
    emb_type = EmbType.SPARSE_SEQ
    def __init__(self,
                 field_name: str, source: FeatureSource,
                 embedding_size=16, num_embeddings=-1,
                 is_string_format: bool = False,
                 separator: str = ","):
        super().__init__(field_name, embedding_size, source)
        self._num_embeddings = num_embeddings
        self.is_string_format = is_string_format
        self.separator = separator
        self.vocab: Dict[str, int] = {}
        if num_embeddings > 0:
            self.is_fitted = True

    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings

    def get_fit_exprs(self) -> List[pl.Expr]:
        # 1. 定义基础展开计算图（此时它是一列很长的、包含各种脏数据的字符串）
        exploded = explode_expr(self.field_name, self.is_string_format, self.separator)
        # 3. 组装终极计算图：过滤 -> 强转 -> 去重 -> 必须 implode() 打包回一行！
        final_expr = (
            exploded.unique().drop_nulls()  # 兜底防御，干掉转换中可能出现的 null
            .implode()  # 【极其关键】将多行合并为一个 List，保证该特征统计结果只占 1 行！
            .alias(self.field_name)
        )
        return [final_expr]

    def parse_fit_result(self, result_df: pl.DataFrame):
        # 提取聚合后的列表
        unique_vals = result_df.get_column(self.field_name).to_list()[0]
        unique_vals = [v for v in unique_vals if v is not None]
        unique_vals = sorted(unique_vals)
        # 统一转为字符串作为字典 Key，规避类型坑。预留 0 为 OOV/Padding
        self.vocab = {str(val): idx + 1 for idx, val in enumerate(unique_vals)}
        self._num_embeddings = len(self.vocab) + 1
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        expr = clear_seq_expr(self.field_name, self.is_string_format, self.separator)
        keys = pl.Series(list(self.vocab.keys()), dtype=pl.Utf8)
        vals = pl.Series(list(self.vocab.values()), dtype=pl.UInt32)

        mapped_expr = (
            expr.list.eval(
                pl.element().cast(pl.Utf8).replace_strict(old=keys, new=vals, default=pl.lit(0, dtype=pl.UInt32)).cast(pl.UInt32)
            )
            .alias(self.field_name)
        )
        return mapped_expr

    def to_dict(self):
        d = super().to_dict()
        d["vocab"] = self.vocab
        return d

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.vocab = state_dict.get("vocab", {})
        self._num_embeddings = state_dict.get("num_embeddings", len(self.vocab) + 1)

    @classmethod
    def from_dict(cls, data):
        obj = cls(data["field_name"], FeatureSource[data["feature_source"]], data["embedding_size"],
                  data["num_embeddings"])
        obj.vocab = data.get("vocab", {})
        obj.is_fitted = data.get("is_fitted", False)
        return obj


class IdSeqEmbSetting(EmbSetting):
    emb_type = EmbType.SPARSE_SEQ

    def __init__(self, field_name: str, seq_len_field_name: str, target_setting: SparseEmbSetting, max_len: int = 50,
                 is_string_format: bool = False, separator: str = ","):
        super().__init__(field_name, -10 ** 6) # 使用负数的原因是告诉你不要尝试访问他的embedding_size属性
        self.seq_len_field_name = seq_len_field_name
        self.target_item_setting = target_setting
        self.max_len = max_len
        self.is_string_format = is_string_format  # True: "1,2,3" | False: [1, 2, 3]
        self.separator = separator
        self.is_fitted = True  # 寄生于 target_item，无需 fit

    @property
    def vocab(self):
        return self.target_item_setting.vocab # 由于SparseEmbSetting.vocab的更新不是原地更新，所以这里要动态引用

    @property
    def num_embeddings(self) -> int:
        return self.target_item_setting.num_embeddings

    def get_fit_exprs(self) -> List[pl.Expr]:
        return []  # 不参与 Fit

    def parse_fit_result(self, result_df: pl.DataFrame):
        pass

    def get_transform_expr(self) -> pl.Expr:
        expr = clear_seq_expr(self.field_name, self.is_string_format, self.separator)
        keys = pl.Series(list(self.vocab.keys()), dtype=pl.Utf8) # 临时冻结在local域
        vals = pl.Series(list(self.vocab.values()), dtype=pl.UInt32) # 临时冻结在local域
        mapped_expr = (
            expr.list.eval(
                pl.element().cast(pl.Utf8).replace_strict(old=keys, new=vals, default=pl.lit(0, dtype=pl.UInt32)).cast(pl.UInt32)
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
        # 本身不持有独立的 vocab，依赖 target_item_setting，因此无需额外恢复数据
        # 只要保证 target_item_setting 被正确 load_state 即可, 不用关心seq_len_field_name这玩意靠传
        self.max_len = state_dict.get("max_len", self.max_len)

    @classmethod
    def from_dict(cls, data):
        # 注意：反序列化时无法直接恢复 target_item_setting，这里依赖 SchemaManager 在加载时做二次绑定
        raise NotImplementedError("ItemSeqEmbSetting 需由 Manager 统一构建依赖关联。")


class MinMaxDenseSetting(EmbSetting):
    emb_type = EmbType.DENSE
    def __init__(self, field_name: str, source: FeatureSource, min_val: float = None, max_val: float = None):
        super().__init__(field_name, 1, source, False)
        self.min_val = min_val
        self.max_val = max_val
        if self.min_val is not None and self.max_val is not None:
            self.is_fitted = True
        else:
            self.is_fitted = False

    def get_fit_exprs(self) -> List[pl.Expr]:
        return [
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().min().alias(f"{self.field_name}_min"),
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().max().alias(f"{self.field_name}_max")
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        # 提取聚合后的 min 和 max
        self.min_val = result_df.get_column(f"{self.field_name}_min").to_list()[0]
        self.max_val = result_df.get_column(f"{self.field_name}_max").to_list()[0]

        # 边界情况处理：你可以后续补充你业务需要的逻辑
        if self.min_val is None or self.max_val is None:
            self.min_val = 0.0
            self.max_val = 1.0  # 全是 Null 的情况预留默认值
        elif self.min_val == self.max_val:
            self.max_val = self.min_val + 1e-6  # 防止 Transform 时除以 0
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        range_val = self.max_val - self.min_val
        expr = (pl.col(self.field_name) - self.min_val) / range_val
        return expr

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
        # 注意：这里去掉了 Sparse 特有的 embedding_size 和 num_embeddings 参数
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            min_val=data.get("min_val"),
            max_val=data.get("max_val")
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj

    @property
    def num_embeddings(self) -> int:
        return -1


