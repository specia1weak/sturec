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

# ==========================================
# 2. 调度层 (Manager Layer) - 计算图统筹
# ==========================================
from pathlib import Path
class SchemaManager:
    ITEM_PROFILE_NAME = "item_profile.parquet"
    USER_PROFILE_NAME = "user_profile.parquet"
    SCHEMA_META_NAME = "feature_meta.json"
    WHOLE_DATA_NAME = "whole_dataframe.parquet"
    def __init__(self, settings_list: List[EmbSetting], work_dir: str,
                 time_field=None, label_field=None, domain_field=None):
        self.settings = settings_list

        ## path & dir
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.meta_filepath = self.work_dir / self.SCHEMA_META_NAME

        ## field_name
        self.uid_field = next(
            (s.field_name for s in self.settings if s.source == FeatureSource.USER_ID),
            None
        )
        self.iid_field =next(
            (s.field_name for s in self.settings if s.source == FeatureSource.ITEM_ID),
            None
        )
        self.time_field = time_field
        self.label_field = label_field
        self.domain_field = domain_field

    def _parse_output_dir(self, output_dir=None):
        if output_dir is None:
            output_dir = self.work_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def prepare_data(self, lazy_df: pl.LazyFrame, output_dir: Union[Path, str]=None, redo=False):
        """
        全自动化流程：自动判断输入文件类型，自动构建计算图
        """
        output_parquet = self._parse_output_dir(output_dir) / self.WHOLE_DATA_NAME
        if self.meta_filepath.exists() and output_parquet.exists() and not redo:
            print("[Info] 检测到已存在元数据和目标文件，跳过处理直接加载 Schema。")
            self.load_schema()
            return pl.scan_parquet(output_parquet)

        print("[*] 正在执行计算图截断与中间态落盘，请稍候...")
        lazy_df = self._make_checkpoint(lazy_df)

        print("[*] 阶段一：构建 Fit 计算图...")
        fit_exprs = []
        for setting in self.settings:
            if not setting.is_fitted:
                fit_exprs.extend(setting.get_fit_exprs())

        if fit_exprs:
            print(f"[*] 启动单次全表扫描获取统计量，包含 {len(fit_exprs)} 个计算节点...")
            fit_result = lazy_df.select(fit_exprs).collect()

            # 分发结果
            for setting in self.settings:
                if not setting.is_fitted:
                    setting.parse_fit_result(fit_result)
            print("[+] 统计量计算与映射表构建完成。")
        else:
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")

        # -----------------------------
        # 阶段二：One-Pass 全量转换 (Transform)
        # -----------------------------
        print("[*] 阶段二：构建 Transform 计算图...")
        transform_exprs = [
            setting.get_transform_expr()
            for setting in self.settings
            if setting.emb_type != EmbType.UNKNOWN
        ]

        print(f"[*] 启动流式转换计算，引擎正在将数据写入 {output_parquet} ...")
        # with_columns() 打包所有转换，sink_parquet() 启动流式按块读写，永远不会 OOM！
        lazy_df.with_columns(transform_exprs) \
            .sink_parquet(output_parquet)

        # -----------------------------
        # 阶段三：元数据落盘
        # -----------------------------
        self.save_schema()
        print("[+] 预处理全流程结束！")
        return pl.scan_parquet(output_parquet)

    def _make_checkpoint(self, lazy_df):
        temp_checkpoint_path = self.work_dir / "temp_checkpoint.parquet"
        lazy_df.sink_parquet(temp_checkpoint_path)
        lazy_df = pl.scan_parquet(temp_checkpoint_path)
        return lazy_df

    def generate_profiles(self, lazy_df: pl.LazyFrame, output_dir: str=None, redo=False):
        """
        利用 FeatureSource 统筹提取并保存静态画像表 (Profile)
        """
        item_profile_path = self._parse_output_dir(output_dir) / self.ITEM_PROFILE_NAME
        user_profile_path = self._parse_output_dir(output_dir) / self.USER_PROFILE_NAME

        print("[*] 阶段三：开始提取静态特征查找表 (Profiles)...")
        # -----------------------------
        # 1. 提取 Item Profile
        # -----------------------------
        # 自动扫描所有注册为 ITEM 的特征
        item_features = [setting.field_name for setting in self.settings
                         if setting.source == FeatureSource.ITEM]
        if self.iid_field and item_features and (not item_profile_path.exists() or redo):
            # 确保主键在提取列表中
            if self.iid_field not in item_features:
                item_features.insert(0, self.iid_field)

            print(f"[*] 正在提取 Item 画像，包含特征: {item_features}")
            (
                lazy_df
                .select(item_features)
                .drop_nulls(subset=[self.iid_field])  # 剔除由于 left join 产生的空物品
                .unique(subset=[self.iid_field])  # 按物品 ID 去重
                .sort(self.iid_field)  # 【极其关键】按 ID 从小到大排序！
                .sink_parquet(item_profile_path)
            )

        # -----------------------------
        # 2. 提取 User Profile (同理)
        # -----------------------------
        user_features = [
            setting.field_name
            for setting in self.settings
            if setting.source == FeatureSource.USER
        ]

        if self.uid_field and user_features and (not user_profile_path.exists() or redo):
            if self.uid_field not in user_features:
                user_features.insert(0, self.uid_field)

            print(f"[*] 正在提取 User 画像，包含特征: {user_features}")
            (
                lazy_df
                .select(user_features)
                .drop_nulls(subset=[self.uid_field])
                .unique(subset=[self.uid_field])
                .sort(self.uid_field)
                .sink_parquet(user_profile_path)
            )
        print("[+] 静态查找表提取完成！")

    def split_dataset(self, lf: pl.LazyFrame, strategy: Literal['loo', 'time', 'sequential_ratio', 'random_ratio'] = "random_ratio",
                      output_dir: str = None, redo: bool = False, **kwargs):
        output_dir = self._parse_output_dir(output_dir)
        StrategyClass = SPLIT_STRATEGIES.get(strategy)
        if not StrategyClass:
            raise ValueError(f"未知的切分策略: {strategy}")
        # 将零散的成员变量和方法打包成标准的上下文对象
        split_ctx = SplitContext(
            uid_field=self.uid_field,
            iid_field=self.iid_field,
            time_field=self.time_field,
            checkpoint_fn=self._make_checkpoint  # 把方法当做变量传递进去 (Callback)
        )
        # 策略实例化与执行
        splitter = StrategyClass(context=split_ctx)
        return splitter.split(lf, output_dir, redo, **kwargs)

    def save_as_dataset(self, train_lf, valid_lf=None, test_lf=None, output_dir: str = None, redo: bool = False):
        """
        如果预设的split_dataset不和你的想法，那么把三个lf传过来，这边直接保存。利好Kuairand等预设test的数据集
        """
        output_dir = self._parse_output_dir(output_dir)
        train_path = output_dir / "train.parquet"
        valid_path = output_dir / "valid.parquet"
        test_path = output_dir / "test.parquet"
        split_paths = str(train_path), str(valid_path), str(test_path)
        if not redo and (train_path.exists() or valid_path.exists() or test_path.exists()):
            return split_paths

        all_lf = [train_lf, valid_lf, test_lf]
        for lf, save_path in zip(all_lf, split_paths):
            if lf is not None:
                lf.sink_parquet(save_path)

        ret_path = [path for lf, path in zip(all_lf, split_paths) if lf is not None]
        return tuple(ret_path)


    def save_schema(self):
        meta_data = [s.to_dict() for s in self.settings]
        with open(self.meta_filepath, 'w') as f:
            json.dump(meta_data, f, indent=4)

    def load_schema(self):
        if not os.path.exists(self.meta_filepath):
            print(f"[Warning] 找不到元数据文件 {self.meta_filepath}，无法加载 schema。")
            return

        with open(self.meta_filepath, 'r', encoding='utf-8') as f:
            meta_data_list = json.load(f)

        # 构建 field_name -> dict 的映射，方便快速查找
        meta_dict = {item["field_name"]: item for item in meta_data_list}

        # 遍历当前注册的 settings，就地回填状态
        for setting in self.settings:
            if setting.field_name in meta_dict:
                saved_state = meta_dict[setting.field_name]

                # 可选：做一个类型安全检查
                if saved_state.get("type") != setting.emb_type.name:
                    raise ValueError(
                        f"特征 {setting.field_name} 的类型不匹配: "
                        f"期望 {setting.emb_type.name}, JSON 中为 {saved_state.get('type')}"
                    )

                setting.load_state(saved_state)
                print(f"[-] 成功恢复特征状态: {setting.field_name}")
            else:
                print(f"[Warning] 特征 {setting.field_name} 在 JSON 元数据中未找到，将保持默认/未拟合状态。")

    def source2emb_size(self, *sources: FeatureSource):
        emb_size = 0
        for setting in self.settings:
            if setting.source in sources:
                emb_size += setting.embedding_size
        return emb_size

    def nums(self, field_name):
        for setting in self.settings:
            if setting.field_name == field_name:
                return setting.num_embeddings
        return None

    def get_setting(self, field_name):
        for setting in self.settings:
            if setting.field_name == field_name:
                return setting
        return None

    def fields(self):
        fields = []
        for setting in self.settings:
            fields.append(setting.field_name)
            if isinstance(setting, IdSeqEmbSetting):
                fields.append(setting.seq_len_field_name)
        for attr_name in ("time_field", "label_field", "domain_field"):
            field_name = getattr(self, attr_name)
            if field_name is not None:
                fields.append(field_name)
        return fields

# ==========================================
# 3. 使用示例 (假设使用 KuaiRand)
# ==========================================
if __name__ == "__main__":
    # 假设特征注册
    user_id_setting = SparseEmbSetting("user_id", FeatureSource.USER)

    # 高级序列特征映射：无论原数据是 "1,5,6" 还是 ["1", "5", "6"]，瞬间处理完毕
    tags = SparseSetEmbSetting(
        field_name="tag",
        source=FeatureSource.ITEM,
        is_string_format=True,
        separator=","
    )

    manager = SchemaManager([user_id_setting, tags])

    from src.convert.kuairand import KuaiRand
    item_lf = pl.scan_csv(KuaiRand.VIDEO_FEATURES)
    inter_lf = pl.scan_csv(KuaiRand.STD_LOG_FORMER_DATA)
    user_lf = pl.scan_csv(KuaiRand.USER_FEATURES)

    whole_lf = inter_lf.join(item_lf, on="video_id", how="left")
    whole_lf = whole_lf.join(user_lf, on="user_id", how="left")
    # 只需要喂入路径，不用 pd.read_csv。即使 kuairand_log.csv 有 50GB 也能在普通笔记本上跑完
    # manager.prepare_data("kuairand_log.csv", "kuairand_processed.parquet")
    manager.prepare_data(whole_lf, "kuairand_processed.parquet", redo=True)
    print("架构编译成功，可供调用。")

    import pandas as pd
    df = pd.read_parquet("kuairand_processed.parquet")
    print(df.head(3).to_string())
