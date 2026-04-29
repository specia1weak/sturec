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

    user_sparse_cols = [
        "user_int_feats_1", "user_int_feats_49", "user_int_feats_50",
        "user_int_feats_51", "user_int_feats_55", "user_int_feats_58",
        "user_int_feats_59", "user_int_feats_82", "user_int_feats_92",
        "user_int_feats_93", "user_int_feats_94", "user_int_feats_95",
        "user_int_feats_96", "user_int_feats_97", "user_int_feats_98",
        "user_int_feats_99", "user_int_feats_100", "user_int_feats_101",
        "user_int_feats_102", "user_int_feats_103", "user_int_feats_104",
        "user_int_feats_105", "user_int_feats_106", "user_int_feats_107",
        "user_int_feats_108", "user_int_feats_109"
    ]

    core_id_cols = [
        "user_id",
        "item_id"
    ]

    item_sparse_cols = [
        "item_int_feats_9", "item_int_feats_13", "item_int_feats_81", "item_int_feats_83"
    ]

    user_mid_sparse_cols = [
        "user_int_feats_48", "user_int_feats_52", "user_int_feats_57", "user_int_feats_86"
    ]

    item_mid_sparse_cols = [
        "item_int_feats_5", "item_int_feats_10", "item_int_feats_84"
    ]

    all_sparse_cols = core_id_cols + user_sparse_cols + item_sparse_cols + user_mid_sparse_cols + item_mid_sparse_cols

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