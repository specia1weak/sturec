# emb/

这是 betterbole 最核心的一层。它决定了“原始字段如何变成可训练张量”。

## 1. 三层结构

```text
Schema 层   -> 每个字段的规则定义
Manager 层  -> 规则编排、拟合、转换、落盘
Embedding 层 -> 在 forward 时按规则取 tensor / 查表
```

## 2. Schema 层

源码在 [`src/betterbole/emb/schema`](../../src/betterbole/emb/schema)。

### `EmbType`

```text
UNKNOWN
SPARSE
MULTI_SPARSE
QUANTILE
SPARSE_SEQ
SPARSE_SET
DENSE
VECTOR_DENSE
DENSE_SEQ
SEQUENCE
```

### `SeqGroupConfig`

```python
SeqGroupConfig(
    group_name: str,
    seq_len_field_name: str,
    max_len: int,
    padding_side: "left" | "right" = "right",
    time_field_name: Optional[str] = None,
)
```

### `EmbSetting`

所有 setting 的抽象基类，定义了这条生命周期：

```text
get_fit_exprs()
parse_fit_result()
get_transform_expr()
get_formatters()
compute_tensor()
```

并提供这些通用属性：

- `field_name`
- `embedding_dim`
- `source`
- `padding_zero`
- `use_oov`
- `vocab`
- `oov_idx`
- `num_embeddings`

### `SparseEmbSetting`

```python
SparseEmbSetting(
    field_name,
    source,
    embedding_dim=16,
    padding_zero=True,
    use_null=True,
    fill_null="null",
    min_freq=1,
    use_oov=True,
)
```

要点：

- 会统计词频。
- 低于 `min_freq` 的值不会进入词表。
- 支持 `fill_null="null" | "oov" | "zero" | None`。
- `compute_tensor()` 通过 `nn.Embedding` 查表。

### `MultiSparseSetting`

用于 list / 分隔字符串表示的多值特征。

```python
MultiSparseSetting(
    field_name,
    source,
    embedding_dim=16,
    max_tag_len=5,
    is_string_format=False,
    separator=",",
    padding_zero=True,
    min_freq=1,
    use_oov=True,
    agg="sum",
)
```

要点：

- 支持字符串和 list 两种输入格式。
- 会先清洗、截断，再做词表映射。
- `agg="sum"` 时直接求和；`agg="mean"` 时会做 mean pooling。

### `QuantileEmbSetting`

把连续值分位数离散化后再查表。

### `MinMaxDenseSetting`

对数值做 min-max 归一化，**不建 embedding table**。

### `VectorDenseSetting`

固定长度向量特征，原始值会保留为 list，真正的 padding / truncation 在 formatter 阶段处理。

### `SequenceSetting`

历史序列 setting。它复用 `element_setting` 的词表，不自己再建一套。

关键约束：

- `from_dict()` 没有实现，说明它不能完全靠 JSON 重建。
- 输出字段通常包括：
  - 序列本体
  - `seq_len_field_name`
  - 可选 `time_field_name`

## 3. Manager 层

源码在 [`src/betterbole/emb/manager.py`](../../src/betterbole/emb/manager.py)。

### 初始化

```python
manager = SchemaManager(
    settings_list=settings,
    work_dir="./workspace/demo",
    time_field="timestamp",
    label_fields="label",
    domain_fields="domain_id",
)
```

### 真实会发生什么

- `uid_field` 会自动从 `FeatureSource.USER_ID` 的 setting 找。
- `iid_field` 会自动从 `FeatureSource.ITEM_ID` 的 setting 找。
- `fit()` 会把 schema 存成 `feature_meta.json`。
- `transform()` 只负责套用已经固化的表达式。

### 常用方法

- `fit(train_raw_lf, low_memory=False)`
- `transform(raw_lf)`
- `prepare_data(lazy_df, output_dir=None, redo=False)`，不推荐作为主路径
- `split_dataset(lf, strategy="random_ratio", **kwargs)`
- `save_as_dataset(train_lf, valid_lf, test_lf, output_dir=None, redo=False)`
- `save_schema()`
- `load_schema()`
- `generate_profiles(lazy_df, output_dir=None, redo=False)`
- `make_checkpoint(lf, file_name="custom_checkpoint.parquet", redo=True, sort_by=None, descending=False)`
- `fields()`
- `get_setting(field_name)`
- `source2emb_dim(*sources)`

### 切分策略

`split_dataset()` 支持：

- `loo`
- `time`
- `sequential_ratio`
- `random_ratio`

对应配置类分别是：

- `LooConfig(k_core=3)`
- `SequentialRatioConfig(train_ratio=0.8, valid_ratio=0.1)`
- `TimeSplitConfig(valid_start, test_start)`
- `RandomRatioConfig(train_ratio=0.8, valid_ratio=0.1, group_by=None)`

### 一个重要坑

`fit()` 发现 `feature_meta.json` 已存在时，会直接加载旧 schema 并跳过重新拟合。
如果你修改了 setting 定义，但还在用同一个 `work_dir`，很容易以为代码生效了，实际上拿到的是旧结果。

## 4. Embedding 层

源码在 [`src/betterbole/emb/emblayer.py`](../../src/betterbole/emb/emblayer.py)。

### `RecEmbedding`

对 `nn.Embedding` 的轻量包装，支持初始化方式和 `reinitialize()`。

### `BoleEmbLayer`

按 `FeatureSource` 分组，把 `EmbSetting.compute_tensor()` 逐个拼起来。

### `SideEmb`

旧式“按 source 取特征”封装，现在更多时候直接用 `OmniEmbLayer`。

### `EmbView`

只是一个轻量视图对象，不持有参数，负责把调用转发给 `OmniEmbLayer`。

### `SeqGroupView`

用来把某个序列组里的：

- 序列本体
- 目标特征
- 序列长度
- 可选时间序列

一次性取出来。

### `OmniEmbLayer`

这是当前最重要的类。

```python
omni = OmniEmbLayer(manager=manager)
# 或
omni = OmniEmbLayer(emb_settings=settings)
```

#### 预置视图

- `whole`
- `whole_without_domain`
- `user_all`
- `item_all`
- `inter`
- `inter_without_domain`
- `user_id`
- `item_id`
- `domain`
- `domain_id`

#### 调用方式

```python
emb = omni.whole(interaction)
emb_dict = omni.whole(interaction, split_by="source")
emb_by_name = omni.whole(interaction, split_by="name")
```

#### 过滤规则

- `target_sources` 先筛 source。
- `include_fields` / `exclude_fields` 再按字段过滤。
- 序列 setting 默认不会混进 `split_by="none"` 的拼接结果，除非你显式 include 它。

#### 输出

- `split_by="none"` -> `Tensor[B, D]`
- `split_by="source"` -> `Dict[FeatureSource, Tensor[B, D_i]]`
- `split_by="name"` -> `Dict[str, Tensor[B, D_i]]`

#### 额外功能

`reinitialize_large_vocab_embeddings(vocab_size_threshold, ...)` 可以把词表过大的 embedding 重新初始化，但这是一个显式 opt-in 的工具，不在默认训练流里自动执行。
