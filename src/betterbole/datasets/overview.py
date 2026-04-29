# ==========================================
# 一、 通用信息打印函数
# ==========================================
import polars as pl


def get_general_info(df: pl.DataFrame) -> dict:
    """
    返回包含行数、列名、类型、不重复数量、空值数量以及数值类型五数概括的字典。
    """
    info = {
        "rows": df.height,
        "cols": {}
    }

    for col_name in df.columns:
        series = df[col_name]
        dtype = series.dtype

        # 获取基础统计信息
        n_unique = series.n_unique()
        n_nulls = series.null_count()  # 新增：计算空值数量

        col_info = {
            "type": str(dtype),
            "n_unique": n_unique,
            "null_count": n_nulls  # 新增：存入字典
        }

        # 判断是否为数值类型，如果是则计算五维数据 (Min, Q1, Median, Q3, Max)
        if dtype.is_numeric():
            # 剔除空值以进行准确计算
            s_clean = series.drop_nulls()
            if len(s_clean) > 0:
                col_info["5_number_summary"] = {
                    "min": s_clean.min(),
                    "q1": s_clean.quantile(0.25),
                    "median": s_clean.median(),
                    "q3": s_clean.quantile(0.75),
                    "max": s_clean.max()
                }
            else:
                col_info["5_number_summary"] = "All Nulls"

        info["cols"][col_name] = col_info

    return info


# ==========================================
# 二、 分组统计函数
# ==========================================
def get_group_stats(df: pl.DataFrame, col_name: str) -> dict:
    """
    传入列名，返回该列各组的名称及组内数量。如果列不存在则报错返回。
    """
    if col_name not in df.columns:
        return {"error": f"列名 '{col_name}' 不存在于数据集中。"}

    # 使用 value_counts() 获取分组及数量
    vc = df[col_name].value_counts()

    # vc 通常包含两列：[col_name, "count"]
    count_col_name = vc.columns[1]

    # 转换为字典 {组名: 数量}
    keys = vc[col_name].to_list()
    values = vc[count_col_name].to_list()

    return dict(zip(keys, values))


# ==========================================
# 三、 Head(3) 完整视图函数
# ==========================================
def get_head_info(df: pl.DataFrame) -> dict:
    """
    以字典方式返回前3行完整数据（防截断遮罩），并同时附带字段的Type信息。
    """
    head_df = df.head(3)
    # as_series=False 会将数据直接转换为 Python 原生的 list，不会被终端打印遮罩
    data_dict = head_df.to_dict(as_series=False)

    result = {}
    for col_name in head_df.columns:
        result[col_name] = {
            "type": str(head_df[col_name].dtype),
            "data": data_dict[col_name]
        }

    return result