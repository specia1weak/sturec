# 📊 数据模块 (data/)

> **层级**: L2 (数据与嵌入层)
>
> 依赖 L1 (`Interaction`、`FeatureSource`)。为 L4 (模型层) 提供格式化后的 Tensor 数据。
>
> ```
> Parquet → DataScanner → DataTransformer → ShuffleBuffer → TensorFormatter → Interaction
> ```

## 流式数据管线架构

```
源数据 (Parquet / CSV)
    │
    ▼
DataScanner ─────────── 流式读取，自动分片
    │
    ▼
DataTransformer ─────── (可选) SchemaManager.transform
    │
    ▼
ShuffleBuffer ───────── 大容量 Buffer + Shuffle
    │
    ▼
TensorFormatter ─────── 每列按规则转 Tensor
    │
    ▼
Interaction ─────────── 模型消费
```

---

## DataScanner — [`data/dataset.py`](../../src/betterbole/data/dataset.py)

扫描 Parquet 文件或 Polars LazyFrame，分片产出 `pyarrow.Table`。

```python
scanner = DataScanner(
    source="path/to/data.parquet",   # str / Path / list[str] / pl.LazyFrame
    read_cols=["user_id", "item_id"], # 可选，只读指定列
    filter_expr=pl.col("label") > 0, # 可选，过滤表达式
)

for table in scanner.iter_batches(
    batch_size=4096,
    worker_id=0,
    num_workers=1,
):
    # table: pyarrow.Table
```

**核心逻辑**：
- 当 `parquet_paths` 不为 None 且无 filter 时，使用 PyArrow 底层迭代器（最高效）
- 否则回退到 Polars `collect_batches()` Lazy 模式
- 多 Worker 下自动按文件或行号分片

---

## DataTransformer — [`data/dataset.py`](../../src/betterbole/data/dataset.py)

将原始宽表实时 transform 为编码宽表。

```python
transformer = DataTransformer(
    manager=manager,                    # SchemaManager
    preprocess_exprs=[...],             # 可选预处理表达式
    filter_expr=None,                   # 可选过滤
    output_col_names=manager.fields(),  # 输出列
)

transformed_table = transformer(raw_table)  # pa.Table → pa.Table
```

---

## ShuffleBuffer — [`data/dataset.py`](../../src/betterbole/data/dataset.py)

大容量 Shuffle Buffer：

```python
buffer = ShuffleBuffer(
    capacity=2_000_000,   # 积累多少行后才开始产出 Batch
    batch_size=4096,      # 每个 Batch 大小
    shuffle=True,         # 是否打乱
    drop_last=True,       # 是否丢弃最后一个不完整 Batch
)

# 数据入队
buffer.push(table)

# 产出 Batch（当积累达到 capacity 时）
for batch_dict in buffer.yield_batches():
    ...

# 清空剩余数据
for batch_dict in buffer.flush():
    ...
```

---

## PipelineStreamDataset — [`data/dataset.py`](../../src/betterbole/data/dataset.py)

完整的流式 Dataset，继承 `torch.utils.data.IterableDataset`：

```python
dataset = PipelineStreamDataset(
    scanner=scanner,
    formatter=formatter,      # TensorFormatter
    transformer=transformer,  # DataTransformer (可选)
    buffer=buffer,            # ShuffleBuffer (可选)
)
```

自动支持 `DataLoader(num_workers>1)`，每个 Worker 自动分片。

---

## ParquetStreamDataset — [`data/dataset.py`](../../src/betterbole/data/dataset.py)

最常用的 Dataset（已编码 Parquet，无需实时 transform）：

```python
ds = ParquetStreamDataset(
    parquet_path="train.parquet",
    manager=manager,
    batch_size=4096,
    shuffle=True,
    drop_last=True,
    shuffle_buffer_size=2_000_000,
)

for interaction in ds:
    # interaction: Interaction (dict of Tensor)
    ...
```

---

## RawParquetStreamDataset — [`data/dataset.py`](../../src/betterbole/data/dataset.py)

读取原始 Parquet + 实时 transform：

```python
ds = RawParquetStreamDataset(
    parquet_path="train_raw.parquet",
    manager=manager,
    batch_size=4096,
    shuffle=True,
    raw_preprocess_exprs=[...],
    raw_filter_expr=None,
)
```

---

## Padding / Formatter — [`data/padding.py`](../../src/betterbole/data/padding.py)

### `ColumnFormatter` 体系

| Formatter | 用途 |
|-----------|------|
| `IntFormatter` | 离散 ID → `torch.long` |
| `DenseFormatter` | 浮点数 → `torch.float32` |
| `VectorDenseFormatter` | 定长向量 → `torch.float32` |
| `PaddedIntSequenceFormatter` | 定长整型序列（右侧/左侧补齐） |
| `PaddedFloatSequenceFormatter` | 定长浮点序列 |
| `PaddedNestedSequenceFormatter` | 嵌套序列（如 tag 序列的序列） |
| `FallbackFormatter` | 自动类型推断 |

### `FeatureContext`

决定每列的读入和输出列名：

```python
context = FeatureContext.from_manager(manager)       # 已编码数据
context = FeatureContext.from_raw_manager(manager)   # 原始数据（读入列=raw，输出列=encoded）
```

### `TensorFormatter`

```python
formatter = TensorFormatter(context)
interaction = formatter.format(batch_dict)  # Dict[str, np.ndarray] → Interaction
```
