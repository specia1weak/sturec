from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union

import polars as pl


@dataclass(frozen=True)
class SplitContext:
    uid_field: Optional[str]
    iid_field: Optional[str]
    time_field: Optional[str]
    checkpoint_fn: Callable[[pl.LazyFrame], pl.LazyFrame]


@dataclass(frozen=True)
class LooConfig:
    k_core: int = 3


@dataclass(frozen=True)
class SequentialRatioConfig:
    train_ratio: float = 0.8
    valid_ratio: float = 0.1


@dataclass(frozen=True)
class TimeSplitConfig:
    valid_start: Any
    test_start: Any


@dataclass(frozen=True)
class RandomRatioConfig:
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    group_by: Optional[str] = None


SplitConfig = Union[LooConfig, SequentialRatioConfig, TimeSplitConfig, RandomRatioConfig]
ConfigT = TypeVar("ConfigT")


class BaseSplitStrategy(Generic[ConfigT], ABC):
    config_type: Type[ConfigT]

    def __init__(self, context: SplitContext, config: ConfigT):
        self.ctx = context
        self.config = config

    @abstractmethod
    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        pass


class LooSplitStrategy(BaseSplitStrategy[LooConfig]):
    config_type = LooConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool):
        print(f"[*] 正在执行 LOO 切分，K-core={self.config.k_core}...")
        if self.ctx.time_field is None:
            raise RuntimeError("执行 loo 切分必须在 SplitContext 中指定 time_field！")

        processed_lf = lf.with_columns(
            [
                pl.count(self.ctx.iid_field).over(self.ctx.uid_field).alias("user_seq_len"),
                pl.col(self.ctx.time_field)
                .rank(descending=True, method="ordinal")
                .over(self.ctx.uid_field)
                .alias("reverse_rank"),
            ]
        ).filter(pl.col("user_seq_len") >= self.config.k_core)

        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)
        drop_cols = ["user_seq_len", "reverse_rank"]
        train_lf = checkpoint_lf.filter(pl.col("reverse_rank") >= 3).select(pl.all().exclude(drop_cols))
        valid_lf = checkpoint_lf.filter(pl.col("reverse_rank") == 2).select(pl.all().exclude(drop_cols))
        test_lf = checkpoint_lf.filter(pl.col("reverse_rank") == 1).select(pl.all().exclude(drop_cols))
        return train_lf, valid_lf, test_lf


class SequentialRatioStrategy(BaseSplitStrategy[SequentialRatioConfig]):
    config_type = SequentialRatioConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        if self.ctx.time_field is None:
            raise RuntimeError("执行 sequential_ratio 切分必须在 SplitContext 中指定 time_field！")

        train_ratio = self.config.train_ratio
        valid_ratio = self.config.valid_ratio
        test_ratio = 1 - train_ratio - valid_ratio
        print(
            f"[*] 正在执行顺序时序比例 (Sequential Ratio) 切分，比例: "
            f"{train_ratio}:{valid_ratio}:{test_ratio:.2f}..."
        )

        time_col = self.ctx.time_field
        time_count_lf = (
            lf.group_by(time_col)
            .len()
            .sort(time_col)
            .with_columns(pl.col("len").cum_sum().alias("cum_rows"))
        )
        total_rows = time_count_lf.select(pl.col("len").sum().alias("total_rows")).collect(engine="streaming").item()
        train_cut = int(total_rows * train_ratio)
        valid_cut = train_cut + int(total_rows * valid_ratio)

        boundary_df = (
            time_count_lf.select(
                [
                    pl.col(time_col).filter(pl.col("cum_rows") >= train_cut).first().alias("valid_start"),
                    pl.col(time_col).filter(pl.col("cum_rows") >= valid_cut).first().alias("test_start"),
                ]
            ).collect(engine="streaming")
        )
        valid_start = boundary_df["valid_start"][0]
        test_start = boundary_df["test_start"][0]

        train_lf = lf.filter(pl.col(time_col) < valid_start)
        valid_lf = lf.filter((pl.col(time_col) >= valid_start) & (pl.col(time_col) < test_start))
        test_lf = lf.filter(pl.col(time_col) >= test_start)
        return train_lf, valid_lf, test_lf


class TimeSplitStrategy(BaseSplitStrategy[TimeSplitConfig]):
    config_type = TimeSplitConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        if self.ctx.time_field is None:
            raise RuntimeError("执行 time 切分必须在 SplitContext 中指定 time_field！")

        print(
            f"[*] 正在执行绝对时间阈值 (Time) 切分，"
            f"train_end={self.config.valid_start}, valid_end={self.config.test_start}..."
        )
        time_expr = pl.col(self.ctx.time_field)
        train_lf = lf.filter(time_expr < self.config.valid_start)
        valid_lf = lf.filter((time_expr >= self.config.valid_start) & (time_expr < self.config.test_start))
        test_lf = lf.filter(time_expr >= self.config.test_start)
        return train_lf, valid_lf, test_lf


class RandomRatioStrategy(BaseSplitStrategy[RandomRatioConfig]):
    config_type = RandomRatioConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        train_ratio = self.config.train_ratio
        valid_ratio = self.config.valid_ratio
        group_by = self.config.group_by

        group_msg = f"按 '{group_by}' 分组" if group_by else "全局"
        print(
            f"[*] 正在执行{group_msg}随机比例 (Random Ratio) 切分，"
            f"比例: {train_ratio}:{valid_ratio}:{1 - train_ratio - valid_ratio:.2f}..."
        )

        len_expr = pl.len().over(group_by) if group_by else pl.len()
        row_idx_expr = pl.int_range(0, pl.len()).shuffle(seed=42)
        if group_by:
            row_idx_expr = row_idx_expr.over(group_by)

        train_idx_expr = (len_expr * train_ratio).cast(pl.Int64)
        valid_idx_expr = train_idx_expr + (len_expr * valid_ratio).cast(pl.Int64)
        processed_lf = lf.with_columns(
            row_idx_expr.alias("row_idx"),
            train_idx_expr.alias("train_idx"),
            valid_idx_expr.alias("valid_idx"),
        )

        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)
        temp_cols = ["row_idx", "train_idx", "valid_idx"]
        train_lf = checkpoint_lf.filter(pl.col("row_idx") < pl.col("train_idx")).drop(temp_cols)
        valid_lf = checkpoint_lf.filter(
            (pl.col("row_idx") >= pl.col("train_idx")) & (pl.col("row_idx") < pl.col("valid_idx"))
        ).drop(temp_cols)
        test_lf = checkpoint_lf.filter(pl.col("row_idx") >= pl.col("valid_idx")).drop(temp_cols)
        return train_lf, valid_lf, test_lf


SPLIT_STRATEGIES: Dict[str, Type[BaseSplitStrategy]] = {
    "loo": LooSplitStrategy,
    "time": TimeSplitStrategy,
    "sequential_ratio": SequentialRatioStrategy,
    "random_ratio": RandomRatioStrategy,
}

SPLIT_CONFIG_TYPES: Dict[str, Type[Any]] = {
    "loo": LooConfig,
    "time": TimeSplitConfig,
    "sequential_ratio": SequentialRatioConfig,
    "random_ratio": RandomRatioConfig,
}


def build_split_config(strategy: str, **kwargs) -> SplitConfig:
    config_cls = SPLIT_CONFIG_TYPES.get(strategy)
    if config_cls is None:
        raise ValueError(f"未知的切分策略: {strategy}")

    allowed_fields = {item.name for item in fields(config_cls)}
    unknown_fields = sorted(set(kwargs) - allowed_fields)
    if unknown_fields:
        raise TypeError(
            f"切分策略 '{strategy}' 不接受参数: {unknown_fields}。"
            f"允许参数为: {sorted(allowed_fields)}"
        )

    return config_cls(**kwargs)


def create_split_strategy(
    strategy: str,
    context: SplitContext,
    config: Optional[SplitConfig] = None,
    **kwargs,
) -> BaseSplitStrategy:
    strategy_cls = SPLIT_STRATEGIES.get(strategy)
    if strategy_cls is None:
        raise ValueError(f"未知的切分策略: {strategy}")

    if config is None:
        config = build_split_config(strategy, **kwargs)
    elif kwargs:
        raise TypeError("请在 split_dataset 中二选一：传入显式 config，或传入兼容 kwargs。")

    if not isinstance(config, strategy_cls.config_type):
        raise TypeError(
            f"切分策略 '{strategy}' 期望配置类型为 {strategy_cls.config_type.__name__}，"
            f"实际得到 {type(config).__name__}"
        )

    return strategy_cls(context=context, config=config)
