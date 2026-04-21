from typing import Tuple

import polars as pl

from betterbole.utils.sequential import extract_history_sequences


def generate_hybrid_splits_polars(
        whole_lf: pl.LazyFrame,
        min_interaction_len: int = 5,
        max_seq_len: int = 20
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    基于 Polars LazyFrame 执行 Hybrid 混合模式切分
    """

    # ==========================================
    # 1. 预处理：排序、计算用户序列长度与切分边界
    # ==========================================
    # 全局排序，确保时间线正确
    lf = whole_lf.sort(["user", "time"])

    # 计算用户总交互数，并为每条记录分配自增行号(0-indexed)
    lf = lf.with_columns(
        pl.len().over("user").alias("user_total_len"),
        pl.int_range(0, pl.len()).over("user").alias("user_row_idx")
    )

    # 过滤短序列用户
    lf = lf.filter(pl.col("user_total_len") >= min_interaction_len)

    # 严格按照你的向下取整逻辑计算边界
    # 在 Polars 中 cast(pl.Int64) 相当于 int() 的截断操作
    lf = lf.with_columns(
        (pl.col("user_total_len") * 0.8).cast(pl.Int64).alias("train_end"),
        (pl.col("user_total_len") * 0.9).cast(pl.Int64).alias("val_end")
    )

    # ==========================================
    # 2. 划定 Split 标签并生成基础的 PLE 数据集
    # ==========================================
    lf = lf.with_columns(
        pl.when(pl.col("user_row_idx") < pl.col("train_end")).then(pl.lit("train"))
        .when(pl.col("user_row_idx") < pl.col("val_end")).then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )

    # 基础列。保留调用方预先挂载的序列/画像列，避免 cross-domain 脚本
    # 在 split 后丢掉 items_seq、seq_len 等模型输入。
    base_cols = ["user", "item", "domain_indicator", "label", "time"]
    helper_cols = {
        "user_total_len",
        "user_row_idx",
        "train_end",
        "val_end",
        "split",
    }
    extra_cols = [
        col
        for col in lf.collect_schema().names()
        if col not in set(base_cols) and col not in helper_cols
    ]
    ple_cols = base_cols + extra_cols

    train_ple = lf.filter(pl.col("split") == "train").select(ple_cols)
    val_ple = lf.filter(pl.col("split") == "val").select(ple_cols)
    test_ple = lf.filter(pl.col("split") == "test").select(ple_cols)

    # ==========================================
    # 3. 构造序列数据 (train_samples)
    # ==========================================
    # 从 Train 集中提取 Label=1 的正样本
    pos_train_lf = train_ple.filter(pl.col("label") == 1)

    # 计算 is_overlapping (用户在正样本中的 domain 是否 >= 2)
    pos_train_lf = pos_train_lf.with_columns(
        (pl.col("domain_indicator").n_unique().over("user") >= 2).alias("is_overlapping")
    )

    # 调用你提供的函数提取多特征序列
    feature_mapping = {
        "item": "item_seq",
        "domain_indicator": "domain_seq"
    }

    train_seq_lf = extract_history_sequences(
        lf=pos_train_lf,
        max_seq_len=max_seq_len,
        user_col="user",
        time_col="time",
        feature_mapping=feature_mapping,
        seq_len_col="seq_len"
    )

    # 按照原代码逻辑：过滤掉前3个正样本，从第4个开始（索引 >= 3）作为 Target
    train_seq_lf = train_seq_lf.with_columns(
        pl.int_range(0, pl.len()).over("user").alias("pos_row_idx")
    ).filter(
        pl.col("pos_row_idx") >= 3
    )

    # 整理最终格式输出
    train_samples = train_seq_lf.select([
        "user",
        pl.col("item"),
        pl.col("domain_indicator"),
        "is_overlapping",
        "time",
        "item_seq",
        "domain_seq",
        "seq_len"
    ])

    return train_samples, train_ple, val_ple, test_ple
