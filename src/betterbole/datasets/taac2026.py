from betterbole.datasets.base import DatasetBase
import polars as pl
"""
### User int Features (46 columns)

- `user_int_feats_{1,3,4,48-59,82,86,92-109}`: Scalar `int64`, total 35 columns.
- `user_int_feats_{15, 60, 62-66, 80, 89-91}`: Array `list<int64>`, total 11 columns.


### User Dense Features (10 columns)

- `user_dense_feats_{61-66, 87, 89-91}`: Array `list<float>`, total 10 columns.


### Item int Features (14 columns)

- `item_int_feats_{5-10, 12-13, 16, 81, 83-85}`: Scalar `int64`, total 13 columns.
- `item_int_feats_{11}`: Array `list<int64>`, total 1 column.


### Domain Sequence Features (45 columns)

`list<int64>` sequences from 4 behavioral domains:

- `domain_a_seq_{38-46}`: 9 columns
- `domain_b_seq_{67-79, 88}`: 14 columns
- `domain_c_seq_{27-37, 47}`: 12 columns
- `domain_d_seq_{17-26}`: 10 columns
"""

"""
统计发现：
1. user_int_feats_{1,49,50,58,92-109} 在sample1000中数量小于10
2. user_int_feats_{55,59,82,93} 出现数量小于50
3. user_int_feats_51 最小值 40，最大值 150，但 n_unique 只有 6，且 Q1、Median、Q3 都是 56。说明绝大部分用户的值都是 56，只有极少数离群点。建议作为 Sparse 对待
4. null占比极高，大于80%的：user_int_feats_101, user_int_feats_100, item_int_feats_83

3. item_int_feats_{9,13,81,83} 出现次数小于等于25
4. label_type{1,2}
"""

class _TAACDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "TAAC2026"
    WHOLE = BASE_DIR / "demo_1000.parquet"

    # ==========================================
    # 1. 核心标识与目标 (Base & Target)
    # ==========================================
    # user_id 绝对不能进模型；label_type 需要减 1 变 0/1；time 用于算时间差
    col_user_id = "user_id"
    col_item_id = "item_id"
    col_label = "label_type"
    cols_time = ["label_time", "timestamp"]

    # ==========================================
    # 2. 物品侧静态特征 (Item Features)
    # ==========================================
    # 常规离散标量 (适合标准 Embedding)
    item_sparse_cols = [
        f"item_int_feats_{i}" for i in [5, 6, 7, 8, 9, 10, 12, 13, 16, 81, 83, 84, 85]
    ]
    # 多值离散特征 (大概率是Tags，适合 Mean/Sum Pooling)
    item_varlen_sparse_cols = ["item_int_feats_11"]

    # ==========================================
    # 3. 用户侧上下文与画像 (User Features)
    # ==========================================
    # 3.1 单值离散特征 (适合标准 Embedding)
    user_sparse_cols = [
        f"user_int_feats_{i}" for i in [1, 3, 4, *range(48, 60), 82, 86, *range(92, 110)]
    ]

    # 3.2 多值离散特征 (近期Tag/行为集合，适合 Mean Pooling)
    user_varlen_sparse_cols = [
        f"user_int_feats_{i}" for i in [15, 60, 62, 63, 64, 65, 66, 80, 89, 90, 91]
    ]

    # 3.3 预训练稠密向量 (高阶画像，适合直接过 MLP 或 Dense 层降维)
    # 注意：这里修正了你的笔误，它们在日志中是 user_dense_feats
    user_dense_emb_cols = [
        f"user_dense_feats_{i}" for i in [61, 87, 89, 90, 91]
    ]

    # 3.4 历史累积统计值 (绝对数值极大，送入模型前必须做 Log(x+1) 平滑处理)
    user_dense_stat_cols = [
        f"user_dense_feats_{i}" for i in [62, 63, 64, 65, 66]
    ]

    # ==========================================
    # 4. 终身行为序列跨域特征 (Cross-Domain Sequences)
    # ==========================================
    # 使用结构化字典管理，便于给 DIN / SIM 等 Attention 模块传参
    seq_domains = {
        "A": {
            "time": "domain_a_seq_39",
            "id": "domain_a_seq_38",
            "attrs": [f"domain_a_seq_{i}" for i in range(40, 47)]
        },
        "B": {
            "time": "domain_b_seq_67",
            "id": "domain_b_seq_69",
            "attrs": [f"domain_b_seq_{i}" for i in [68, *range(70, 80), 88]]
        },
        "C": {
            "time": "domain_c_seq_27",
            "id": "domain_c_seq_47",
            "attrs": [f"domain_c_seq_{i}" for i in range(28, 38)]
        },
        "D": {
            # D域无明显独立时间戳（或隐含对齐），长达3800，必须做 SIM 截断检索
            "time": "domain_d_seq_26",
            "id": "domain_d_seq_23",
            "attrs": [f"domain_d_seq_{i}" for i in range(17, 27) if i not in (23, 26)]
        }
    }
    high_null_cols = ["user_int_feats_101", "user_int_feats_100", "item_int_feats_83"]
    low_cardinality_cols = [
        f"user_int_feats_{i}" for i in [1, 49, 50, 58, 55, 59, 82, 93, *range(92, 110)]
    ]

    @property
    def WHOLE_LF(self):
        whole_lf = pl.scan_parquet(self.WHOLE)
        whole_lf = whole_lf.with_columns(pl.col("label_type") - 1)
        return whole_lf

TAAC2026Dataset = _TAACDS()

if __name__ == '__main__':
    import polars as pl
    from betterbole.datasets.overview import get_general_info
    lf = pl.scan_parquet(TAAC2026Dataset.WHOLE).collect()
    info = get_general_info(lf)
    from pprint import pprint
    pprint(info)