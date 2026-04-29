from typing import Union, Tuple, Dict

import polars as pl
from typing import Dict, Optional, Any


def extract_history_sequences(
        lf: pl.LazyFrame,
        max_seq_len: int,
        user_col: str,
        time_col: str,
        feature_mapping: Dict[str, str],
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1
) -> pl.LazyFrame:
    """
    严格提取用户历史正向行为序列。避免负样本稀释，且绝对保证无数据泄漏（防穿越）。
    """
    if feature_mapping is None:
        feature_mapping = {"item_id": "item_seq"}

    # 1. 确保全局严格排序，并分配全局连续的物理行号作为“逻辑时钟”
    lf = lf.sort([user_col, time_col])
    lf = lf.with_row_index("_global_row_idx")

    # 2. 剥离出纯正样本的 DataFrame (继承了主表的 _global_row_idx)
    if label_col is not None:
        pos_lf = lf.filter(pl.col(label_col) == positive_label)
    else:
        pos_lf = lf

    # 在纯正样本中，为每个用户打上局部的连续序号 (0, 1, 2...)
    pos_lf = pos_lf.with_columns(
        pl.int_range(0, pl.len(), dtype=pl.UInt32).over(user_col).alias("_pos_idx")
    )

    # 3. 在纯正样本上进行严格的 N 长度开窗
    # 因为表中全都是正样本，所以开窗 max_seq_len 就能拿满 N 个正交互
    agg_exprs = [
        pl.col(src).alias(tgt_seq_name)
        for src, tgt_seq_name in feature_mapping.items()
    ]

    pos_rolling = pos_lf.rolling(
        index_column="_pos_idx",
        group_by=user_col,
        period=f"{max_seq_len}i"
    ).agg(agg_exprs)

    # 将生成的纯正序列和原有的“逻辑时钟”拼起来，且重命名避免冲突
    # pos_state 代表：在这个 _state_row_idx 时刻【之后】，用户的正向历史状态更新为了什么
    pos_state = pos_lf.select([user_col, "_global_row_idx", "_pos_idx"]).join(
        pos_rolling, on=[user_col, "_pos_idx"]
    ).rename({"_global_row_idx": "_state_row_idx"}).drop("_pos_idx")

    # 【关键修复】: 将 state_row_idx 转为有符号整数并严格排序
    pos_state = pos_state.with_columns(
        pl.col("_state_row_idx").cast(pl.Int64)
    ).sort("_state_row_idx")

    # 4. ASOF JOIN 拼回主表 (核心防泄漏机制)
    # 对于主表的每一行，我们要找的是严格发生在它【之前】的最新正向状态
    # 【关键修复】: 左表的 join_idx = 当前行号 - 1，必须先 cast 为 Int64 避免无符号 0 - 1 下溢
    lf = lf.with_columns(
        (pl.col("_global_row_idx").cast(pl.Int64) - 1).alias("_join_idx")
    )

    lf = lf.join_asof(
        pos_state,
        left_on="_join_idx",
        right_on="_state_row_idx",
        by=user_col,
        strategy="backward"  # 寻找 <= _join_idx 的最新状态
    )

    # 5. 清理与填补
    first_seq_col = list(feature_mapping.values())[0]
    drop_cols = ["_global_row_idx", "_join_idx", "_state_row_idx"]

    # 将匹配不到历史状态的 Null 填充为空列表 []
    fill_exprs = [
        pl.col(tgt).fill_null([]) for tgt in feature_mapping.values()
    ]

    lf = (
        lf.with_columns(fill_exprs)
        .with_columns(
            pl.col(first_seq_col).list.len().fill_null(0).cast(pl.UInt32).alias(seq_len_col)
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
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1
) -> pl.LazyFrame:
    return extract_history_sequences(
        lf, max_seq_len, user_col, time_col,
        {item_col: seq_col}, seq_len_col,
        label_col, positive_label
    )

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
            "video_id": ["v_A", "v_C", "v_B", "v_D", "v_E", "v_G", "v_F"],
            "label": [0, 0, 1, 1, 1, 1, 1]
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
            label_col="label",
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