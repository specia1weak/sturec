from typing import Union, Tuple, Dict

import polars as pl
from typing import Dict


def extract_history_sequences(
        lf: pl.LazyFrame,
        max_seq_len: int,
        user_col: str,
        time_col: str,
        feature_mapping: Dict[str, str],
        seq_len_col: str = "seq_len"
) -> pl.LazyFrame:
    """
    提取用户历史行为序列，支持同时提取多个上下文特征序列。
    :param feature_mapping: 字典格式，键为原列名，值为生成的序列列名。
                            例如: {"item_id": "item_seq", "category_id": "cat_seq"}
    """
    if feature_mapping is None:
        feature_mapping = {"item_id": "item_seq"}

    # 1. 确保数据严格按用户和时间排序
    lf = lf.sort([user_col, time_col])
    # 2. 将所有需要提取的特征列都下移一行，并赋予临时列名
    shift_exprs = [
        pl.col(src).shift(1).over(user_col).alias(f"_prev_{src}")
        for src in feature_mapping.keys()
    ]
    lf = lf.with_columns(shift_exprs)
    # 3. 生成全局行号
    lf = lf.with_row_index("_row_idx")
    agg_exprs = [
        pl.col(f"_prev_{src}").drop_nulls().alias(tgt_seq_name)
        for src, tgt_seq_name in feature_mapping.items()
    ]

    # 执行滚动聚合
    rolling_lf = (
        lf.rolling(
            index_column="_row_idx",
            group_by=user_col,
            period=f"{max_seq_len}i"
        )
        .agg(agg_exprs)
    )

    # 5. 清理临时列
    drop_cols = ["_row_idx"] + [f"_prev_{src}" for src in feature_mapping.keys()]

    # 随便取一个生成的序列列来计算序列长度（因为它们是平行且等长的）
    first_seq_col = list(feature_mapping.values())[0]

    lf = (
        lf.join(rolling_lf, on=[user_col, "_row_idx"], how="left")
        .with_columns(
            pl.col(first_seq_col).list.len().fill_null(0).alias(seq_len_col)
        )
        .drop(drop_cols)
    )

    return lf

def extract_history_items(
    lf: pl.LazyFrame,
    max_seq_len: int,
    user_col: str,
    time_col: str,
    item_col: str,
    seq_col: str = "items_seq",
    seq_len_col: str="seq_len"
) -> pl.LazyFrame:
    return extract_history_sequences(lf, max_seq_len, user_col, time_col, {item_col: seq_col}, seq_len_col)


def extract_history_dict(
        *lfs: pl.LazyFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        merge: bool = False
) -> Union[Dict[int, list], Tuple[Dict[int, list], ...]]:
    """
    极速从任意数量的 Polars LazyFrame 中提取 User-History 字典。

    Args:
        *lfs: 任意数量的 pl.LazyFrame (例如 train_lf, valid_lf, test_lf)
        user_col: 用户列的列名
        item_col: 物品列的列名
        merge: 是否将所有输入的 LazyFrame 合并成一个全局静态大字典。
               True -> 返回单个 Dict
               False -> 返回 Tuple[Dict, ...], 与输入的 lf 数量一一对应
    Returns:
        history_dict 或者 (history_dict1, history_dict2, ...)
    """
    # 核心聚合算子：在 Lazy 模式下定义图计算逻辑
    def _build_dict_from_lf(lf: pl.LazyFrame) -> Dict:
        lf_ui = lf.select([user_col, item_col])
        df_agg = lf_ui.group_by(user_col, maintain_order=False).agg(
            pl.col(item_col)
        ).collect()  # 这里才真正触发计算，返回 DataFrame
        return dict(zip(df_agg[user_col].to_list(), df_agg[item_col].to_list()))

    # ================= 分支 1：合并计算 =================
    if merge:
        if not lfs:
            return {}
        selected_lfs = [lf.select([user_col, item_col]) for lf in lfs]
        merged_lf = pl.concat(selected_lfs)
        return _build_dict_from_lf(merged_lf)
    else:
        # 直接利用列表推导式，依次处理返回 Tuple
        return tuple(_build_dict_from_lf(lf) for lf in lfs)


if __name__ == "__main__":
    def main():
        # 1. 构造模拟数据：两个用户，交易时间乱序（测试排序逻辑）
        data = {
            "user_id": [1, 2, 1, 1, 2, 1, 2],
            "timestamp": [100, 102, 101, 103, 200, 202, 201],
            "video_id": ["v_A", "v_C", "v_B", "v_D", "v_E", "v_G", "v_F"]
        }

        lf = pl.LazyFrame(data)

        # 2. 设置最大序列长度
        MAX_SEQ_LEN = 3

        print(f"--- 原始输入 (LazyFrame) ---")
        print(lf.collect())

        # 3. 调用你的提取函数
        # 逻辑预期：对于每个用户，当前行的 items_seq 应该是其【之前】的最多 3 个 video_id
        result_lf = extract_history_sequences(
            lf=lf,
            max_seq_len=MAX_SEQ_LEN,
            user_col="user_id",
            time_col="timestamp",
            feature_mapping={
                "video_id": "history_list"
            }
        )

        # 4. 执行并打印结果
        df_result = result_lf.collect()

        print(f"\n--- 处理后的结果 (max_seq_len={MAX_SEQ_LEN}) ---")
        # 按照用户和时间打印，方便观察序列增长
        print(df_result.sort(["user_id", "timestamp"]))

        # 5. 验证核心逻辑点
        print("\n--- 逻辑校验 ---")
        # 校验用户1的最后一行：时间103，前三行应为 [v_A, v_B, v_C]
        last_user1 = df_result.filter((pl.col("user_id") == 1) & (pl.col("timestamp") == 103))
        print(f"用户1(T=103) 的历史序列: {last_user1.get_column('history_list').to_list()}")

        # 校验用户2的第一行：历史序列应该是空的 []
        first_user2 = df_result.filter((pl.col("user_id") == 2) & (pl.col("timestamp") == 200))
        print(f"用户2(T=200) 的历史序列: {first_user2.get_column('history_list').to_list()}")
    main()