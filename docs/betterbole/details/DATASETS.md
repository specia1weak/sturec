# 🗂️ 数据集模块 (datasets/)

> **层级**: L1 (基础层)
>
> 自包含。为 L2 (数据层) 提供数据集加载入口。

## 内置数据集

| 数据集 | 类 | 源文件 |
|--------|----|--------|
| KuaiRand | `KuaiRand` | [`kuairand.py`](../../src/betterbole/datasets/kuairand.py) |
| Amazon | `Amazon` | [`amazon.py`](../../src/betterbole/datasets/amazon.py) |
| MovieLens | `MovieLens` | [`movielens.py`](../../src/betterbole/datasets/movielens.py) |
| Douban | `Douban` | [`douban.py`](../../src/betterbole/datasets/douban.py) |
| Alibaba CCP | `AliCCP` | [`aliccp.py`](../../src/betterbole/datasets/aliccp.py) |
| Tencent TAAC2026 | `TAAC2026` | [`taac2026.py`](../../src/betterbole/datasets/taac2026.py) |

所有数据集继承自 [`DatasetBase`](../../src/betterbole/datasets/base.py)：

```python
class DatasetBase:
    SYSTEM_DATA_DIR = Path("D:/pyprojects/recommend-study/Datasets/")
```

---

## 数据概览工具 — [`overview.py`](../../src/betterbole/datasets/overview.py)

### `get_general_info(df)`

数据集的通用统计信息：

```python
info = get_general_info(df)
# {
#   "rows": 1000000,
#   "cols": {
#     "user_id": {"type": "Int64", "n_unique": 10000, "null_count": 0,
#                 "5_number_summary": {"min": 1, "q1": 2501, "median": 5000, "q3": 7500, "max": 10000}},
#   }
# }
```

### `get_group_stats(df, col_name)`

分组计数：

```python
stats = get_group_stats(df, "domain_id")
# {"0": 500000, "1": 300000, "2": 200000}
```

### `get_head_info(df)`

前 3 行完整视图（防截断）：

```python
head = get_head_info(df)
# {"user_id": {"type": "Int64", "data": [1, 1, 2]}, ...}
```
