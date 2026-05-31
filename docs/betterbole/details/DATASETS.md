# datasets/

这一层不是下载器，而是“本地数据路径和读取入口”的集合。

## 1. 基础类

源码在 [`src/betterbole/datasets/base.py`](../../src/betterbole/datasets/base.py)。

```python
class DatasetBase:
    SYSTEM_DATA_DIR = Path("D:/pyprojects/recommend-study/Datasets/")
```

这个路径是硬编码的。文档层面要把它看成“数据根目录约定”，不是通用下载逻辑。

## 2. 当前可用的数据集对象

### `MovieLensDataset`

源码在 [`src/betterbole/datasets/movielens.py`](../../src/betterbole/datasets/movielens.py)。

- `MovieLensDataset.USER_FEATURES_DF`
- `MovieLensDataset.ITEM_FEATURES_DF`
- `MovieLensDataset.INTERACTION_DF`

这些属性返回的是 `pandas.DataFrame`。

### `KuaiRandDataset`

源码在 [`src/betterbole/datasets/kuairand.py`](../../src/betterbole/datasets/kuairand.py)。

- `STD_LOG_FORMER_DATA`
- `STD_LOG_FORMER_DATA_P1`
- `STD_LOG_FORMER_DATA_P2`
- `RAND_LOG_FORMER_DATA`
- `USER_FEATURES`
- `VIDEO_FEATURES`

### `AliCCPDataset`

源码在 [`src/betterbole/datasets/aliccp.py`](../../src/betterbole/datasets/aliccp.py)。

- `TRAIN_INTER_LF`
- `TEST_INTER_LF`

属性返回 `pl.LazyFrame`，并在第一次访问时缓存 parquet。

### `DoubanDataset`

源码在 [`src/betterbole/datasets/douban.py`](../../src/betterbole/datasets/douban.py)。

- `MERGED_INTERS_LF`
- `ALL_INTERS_LF`
- `USER_LF`

### `TAAC2026Dataset`

源码在 [`src/betterbole/datasets/taac2026.py`](../../src/betterbole/datasets/taac2026.py)。

- `WHOLE`
- `WHOLE_LF`
- `col_user_id`
- `col_item_id`
- `col_label`
- `cols_time`
- 一组 item / user / dense / sequence 字段名列表

## 3. `overview.py`

源码在 [`src/betterbole/datasets/overview.py`](../../src/betterbole/datasets/overview.py)。

### `get_general_info(df)`

返回行数、列类型、唯一值、空值，以及数值列的五数概括。

### `get_group_stats(df, col_name)`

返回某列每个取值的计数。

### `get_head_info(df)`

返回前 3 行的完整字典视图，适合避免终端截断。

## 4. 文档层面的结论

- `datasets/` 里没有统一的下载/解压接口。
- 不同数据集对象返回的类型不完全一致：有的返回 `DataFrame`，有的返回 `LazyFrame`。
- 旧文档里把这些类写成统一 “Dataset” 基类体系，这是过度抽象，当前源码没有这么整齐。
