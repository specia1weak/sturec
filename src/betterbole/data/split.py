from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable

import polars as pl
from pathlib import Path

@dataclass
class SplitContext:
    uid_field: str
    iid_field: str
    time_field: Optional[str]  # 允许为空，供不需要时间的策略使用
    checkpoint_fn: Callable[[pl.LazyFrame], pl.LazyFrame]  # 注入落盘动作，而不是依赖主类方法

class BaseSplitStrategy(ABC):
    def __init__(self, context: SplitContext):
        self.ctx = context

    @abstractmethod
    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool, **kwargs) -> tuple:
        """所有具体的切分策略都必须实现这个方法"""
        pass



class LooSplitStrategy(BaseSplitStrategy):
    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool, **kwargs):
        k_core = kwargs.get('k_core', 3)
        print(f"[*] 正在执行 LOO 切分，K-core={k_core}...")

        # 直接使用 ctx 里的纯数据字段
        processed_lf = lf.with_columns([
            pl.count(self.ctx.iid_field).over(self.ctx.uid_field).alias("user_seq_len"),
            pl.col(self.ctx.time_field).rank(descending=True, method="ordinal").over(self.ctx.uid_field).alias(
                "reverse_rank")
        ]).filter(
            pl.col("user_seq_len") >= k_core
        )

        # 调用被注入进来的回调函数，不关心它在主类里是怎么实现的
        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)

        train_lf = checkpoint_lf.filter(pl.col("reverse_rank") >= 3).select(
            pl.all().exclude("user_seq_len", "reverse_rank"))
        valid_lf = checkpoint_lf.filter(pl.col("reverse_rank") == 2).select(
            pl.all().exclude("user_seq_len", "reverse_rank"))
        test_lf = checkpoint_lf.filter(pl.col("reverse_rank") == 1).select(
            pl.all().exclude("user_seq_len", "reverse_rank"))

        return train_lf, valid_lf, test_lf


class SequentialRatioStrategy(BaseSplitStrategy):
    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool, **kwargs) -> tuple:
        train_ratio = kwargs.get('train_ratio', 0.8)
        valid_ratio = kwargs.get('valid_ratio', 0.1)

        if self.ctx.time_field is None:
            raise RuntimeError("执行 sequential_ratio 切分必须在 SplitContext 中指定 time_field！")

        print(
            f"[*] 正在执行顺序时序比例 (Sequential Ratio) 切分，比例: {train_ratio}:{valid_ratio}:{1 - train_ratio - valid_ratio:.2f}...")

        # 1. 全局按时间升序排序 (历史在前，未来在后)
        processed_lf = lf.sort(self.ctx.time_field)
        # 2. 调用注入的 checkpoint 动作，落盘锁死物理时间顺序
        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)
        # 3. 添加连续的物理行号
        checkpoint_lf = checkpoint_lf.with_row_index("row_idx")
        # 4. 获取精确切分点 (对 checkpoint_lf 做 count 是瞬间完成的)
        total_rows = checkpoint_lf.select(pl.len()).collect().item()
        train_idx = int(total_rows * train_ratio)
        valid_idx = train_idx + int(total_rows * valid_ratio)

        # 5. 按行号区间过滤，并丢弃辅助的行号列
        train_lf = checkpoint_lf.filter(pl.col("row_idx") < train_idx).drop("row_idx")
        valid_lf = checkpoint_lf.filter((pl.col("row_idx") >= train_idx) & (pl.col("row_idx") < valid_idx)).drop("row_idx")
        test_lf = checkpoint_lf.filter(pl.col("row_idx") >= valid_idx).drop("row_idx")

        return train_lf, valid_lf, test_lf


class TimeSplitStrategy(BaseSplitStrategy):
    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool, **kwargs) -> tuple:
        # 这个策略特有的必要参数
        valid_start = kwargs.get('valid_start')
        test_start = kwargs.get('test_start')

        if valid_start is None or test_start is None:
            raise RuntimeError("执行 time 切分必须在 kwargs 中传入 valid_start 和 test_start！")
        if self.ctx.time_field is None:
            raise RuntimeError("执行 time 切分必须在 SplitContext 中指定 time_field！")
        print(f"[*] 正在执行绝对时间阈值 (Time) 切分，train_end={valid_start}, valid_end={test_start}...")
        time_expr = pl.col(self.ctx.time_field)
        # 除非上游的 lf 经历了极端复杂的 join，通常不用在这层做 checkpoint
        train_lf = lf.filter(time_expr < valid_start)
        valid_lf = lf.filter((time_expr >= valid_start) & (time_expr < test_start))
        test_lf = lf.filter(time_expr >= test_start)

        return train_lf, valid_lf, test_lf


class RandomRatioStrategy(BaseSplitStrategy):
    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool, **kwargs) -> tuple:
        train_ratio = kwargs.get('train_ratio', 0.8)
        valid_ratio = kwargs.get('valid_ratio', 0.1)
        group_by = kwargs.get('group_by', None)

        group_msg = f"按 '{group_by}' 分组" if group_by else "全局"
        print(
            f"[*] 正在执行{group_msg}随机比例 (Random Ratio) 切分，比例: {train_ratio}:{valid_ratio}:{1 - train_ratio - valid_ratio:.2f}...")
        # 1. 构造动态的总长度表达式
        # 如果有 group_by，获取的是该组的总行数；否则是全局总行数
        len_expr = pl.len().over(group_by) if group_by else pl.len()

        # 构造局部/全局随机乱序 ID
        row_idx_expr = pl.int_range(0, pl.len()).shuffle(seed=42)
        if group_by:
            row_idx_expr = row_idx_expr.over(group_by)

        # 2. 使用 cast(pl.Int64) 完全复刻原代码中 int() 的向下取整逻辑
        train_idx_expr = (len_expr * train_ratio).cast(pl.Int64)
        valid_idx_expr = train_idx_expr + (len_expr * valid_ratio).cast(pl.Int64)

        # 3. 将计算好的辅助列合并到数据流中
        processed_lf = lf.with_columns(
            row_idx_expr.alias("row_idx"),
            train_idx_expr.alias("train_idx"),
            valid_idx_expr.alias("valid_idx")
        )

        # 4. 调用 checkpoint (保留你原本的流水线缓存/执行节点)
        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)
        # 5. 根据精准的整数索引进行过滤，最后再抛弃辅助列
        temp_cols = ["row_idx", "train_idx", "valid_idx"]
        train_lf = checkpoint_lf.filter(pl.col("row_idx") < pl.col("train_idx")).drop(temp_cols)
        valid_lf = checkpoint_lf.filter((pl.col("row_idx") >= pl.col("train_idx")) & (pl.col("row_idx") < pl.col("valid_idx"))).drop(temp_cols)
        test_lf = checkpoint_lf.filter(pl.col("row_idx") >= pl.col("valid_idx")).drop(temp_cols)
        return train_lf, valid_lf, test_lf

SPLIT_STRATEGIES = {
    "loo": LooSplitStrategy,
    "time": TimeSplitStrategy,
    "sequential_ratio": SequentialRatioStrategy,
    "random_ratio": RandomRatioStrategy
}