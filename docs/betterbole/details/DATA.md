# data/

`data/` 是把 `LazyFrame / Parquet` 变成 `Interaction` 的流水线层。

## 1. 总流程

```text
Parquet / LazyFrame
  -> DataScanner
  -> (optional) DataTransformer
  -> ShuffleBuffer
  -> TensorFormatter
  -> Interaction
```

## 2. `DataScanner`

源码在 [`src/betterbole/data/dataset.py`](../../src/betterbole/data/dataset.py)。

```python
scanner = DataScanner(
    source="train.parquet",          # str / Path / list[str] / pl.LazyFrame
    read_cols=["user_id", "item_id"],
    filter_expr=pl.col("label") > 0,
)
```

### 行为

- 如果 `source` 是 parquet 路径且没有 `filter_expr`，会优先走 `pyarrow.parquet` 的批读取。
- 如果 `source` 是 `pl.LazyFrame` 或存在过滤条件，会走 Polars lazy pipeline。
- 多 worker 时会自动分片：
  - parquet 路径模式按文件索引切分
  - lazy 模式按行号取模切分

### `iter_batches(batch_size, worker_id=0, num_workers=1)`

返回的是 `pyarrow.Table` 迭代器，不是 tensor。

## 3. `DataTransformer`

```python
transformer = DataTransformer(
    manager=manager,
    preprocess_exprs=[pl.col("date").cast(pl.Utf8).str.to_date()],
    filter_expr=pl.col("label").is_not_null(),
    output_col_names=manager.fields(),
)
```

它做三件事：

1. 把 `pyarrow.Table` 转成 `pl.DataFrame`
2. 可选执行预处理表达式和过滤
3. 调 `manager.transform(...)` 再转回 `pyarrow.Table`

## 4. `ShuffleBuffer`

```python
buffer = ShuffleBuffer(
    capacity=2_000_000,
    batch_size=4096,
    shuffle=True,
    drop_last=True,
)
```

### 行为

- 先积累到 `capacity`，再统一产出 batch。
- batch 维度上的 shuffle 是在 buffer 内做的。
- `yield_batches()` 只有在当前缓存行数达到 `capacity` 时才真正输出。
- `flush()` 用来把最后一段剩余数据吐出来。

## 5. `PipelineStreamDataset`

这是所有流式数据集的基类。

```python
dataset = PipelineStreamDataset(
    scanner=scanner,
    formatter=formatter,
    transformer=transformer,
    buffer=buffer,
)
```

`__iter__()` 会按 worker 自动读取、转换、切 batch、格式化，最终输出 `Interaction`。

## 6. `ParquetStreamDataset`

这是最常用的版本，前提是你已经有编码后的 parquet。

```python
ds = ParquetStreamDataset(
    parquet_path="train.parquet",
    manager=manager,
    batch_size=4096,
    shuffle=True,
    drop_last=True,
)
```

### 关键点

- 默认读取列由 `FeatureContext.from_manager(manager)` 推导。
- `extra_col_names` 和 `extra_col_formatters` 可以补充额外列。
- `batch_size` 只控制 dataset 内部 batch 的大小，不是外层 `DataLoader` 的 batch。

## 7. `RawParquetStreamDataset`

用于“原始 parquet，读取时再 transform”。

```python
ds = RawParquetStreamDataset(
    parquet_path="train_raw.parquet",
    manager=manager,
    batch_size=4096,
    shuffle=True,
    raw_preprocess_exprs=[pl.col("date").cast(pl.Utf8)],
    raw_filter_expr=pl.col("split") == "train",
)
```

### 行为细节

- 如果 `raw_preprocess_exprs` 为空，`raw_filter_expr` 会先在 scanner 阶段生效。
- 如果 `raw_preprocess_exprs` 不为空，过滤会延后到 transformer 阶段。

## 8. `FeatureContext` 和 `TensorFormatter`

源码在 [`src/betterbole/data/padding.py`](../../src/betterbole/data/padding.py)。

### `FeatureContext`

```python
FeatureContext.from_manager(manager)
FeatureContext.from_raw_manager(manager)
```

- `from_manager()` 用于已经编码好的数据。
- `from_raw_manager()` 用于原始数据，需要同时推导读入列和输出列。

### `TensorFormatter`

```python
formatter = TensorFormatter(context)
interaction = formatter.format(batch_dict)
```

它会根据每个字段对应的 `EmbSetting.get_formatters()` 把 numpy / list 转成 tensor，再包成 `Interaction`。

## 9. Formatter 类型

- `DenseFormatter`
- `VectorDenseFormatter`
- `IntFormatter`
- `FallbackFormatter`
- `PaddedIntSequenceFormatter`
- `PaddedFloatSequenceFormatter`
- `PaddedNestedSequenceFormatter`

### 约束

- `VectorDenseFormatter` 会固定到 `dim`，超长截断，短的补 0。
- `PaddedNestedSequenceFormatter` 用于嵌套 list，例如“标签序列的序列”。
- `FallbackFormatter` 只是兜底，不要拿它当正式 schema 规则。
