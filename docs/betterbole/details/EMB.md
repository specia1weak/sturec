# 🧬 嵌入模块 (emb/)

> **层级**: L2 (数据与嵌入层)
>
> 依赖 L1 (`FeatureSource`、`Interaction`)。是 betterbole 最核心的模块。
>
> 为 L4 (模型层) 提供 `OmniEmbLayer` 统一特征注入。

betterbole 的特征工程核心，分为三层：

```
Schema 层 (emb/schema/)    → 每个字段的"规则"定义
Manager 层 (emb/manager/)  → 规则编排与调度
Embedding 层 (emb/emblayer/) → 规则落地为 nn.Embedding 查表
```

---

## 一、Schema 层 — 特征规则

### `EmbSetting` 基类 — [`schema/base.py`](../../src/betterbole/emb/schema/base.py)

所有特征规则的抽象基类，定义了标准生命周期：

```
get_fit_exprs()           → 1. 生成统计表达式
parse_fit_result()        → 2. 从统计结果恢复参数
get_transform_expr()      → 3. 生成变换表达式
get_formatters()          → 4. 定义该列如何转 Tensor
compute_tensor()          → 5. 前向：从 Interaction 获取 Embedding
```

### `EmbType` — 特征类型枚举

| 类型 | 说明 |
|------|------|
| `SPARSE` | 单值离散特征 (user_id, item_id) |
| `MULTI_SPARSE` | 多值集合特征 (tags, multi-hot 类目) |
| `QUANTILE` | 连续值分桶离散化 |
| `DENSE` | 连续浮点特征 |
| `VECTOR_DENSE` | 定长向量特征 |
| `SEQUENCE` | 历史行为序列 |
| `SPARSE_SEQ` | 稀疏序列（兼容旧版） |
| `DENSE_SEQ` | 稠密序列（兼容旧版） |
| `SPARSE_SET` | 稀疏集合（兼容旧版 `MultiSparseSetting`） |

---

### `SparseEmbSetting` — [`schema/categorical.py`](../../src/betterbole/emb/schema/categorical.py)

**单值离散特征**，最常用。

```python
SparseEmbSetting(
    field_name="user_id",
    source=FeatureSource.USER_ID,
    embedding_dim=16,
    padding_zero=True,     # 是否保留 0 作为 Padding
    min_freq=1,            # 最少出现次数过滤低频
    use_oov=True,          # 是否启用 OOV
)
```

**核心逻辑**：
- `fit()` 统计各值频次，低于 `min_freq` 的归为 OOV
- `transform()` 将原始值映射为 `[1, vocab_size]` 的编码 ID
- `oov_idx = vocab_size + 1`（如果 `use_oov=True`)
- `padding_idx = 0`（如果 `padding_zero=True`）

**Tensor 计算**：`nn.Embedding(interaction[field])` → `[B, D]`

---

### `MultiSparseSetting` — [`schema/categorical.py`](../../src/betterbole/emb/schema/categorical.py)

**多值集合特征** (tags、多值类目等)。

```python
MultiSparseSetting(
    field_name="tag",
    source=FeatureSource.ITEM,
    embedding_dim=16,
    max_tag_len=5,          # 每个集合最多保留的元素数
    is_string_format=False, # 输入是否为字符串 "a,b,c"
    separator=",",          # 字符串分隔符
    agg="sum",              # 聚合方式: sum / mean
    min_freq=1,
    use_oov=True,
)
```

**核心逻辑**：
- 支持字符串格式和 list 格式两种输入
- 先 explode 再按频次建词表
- transform 时逐元素映射到词表 ID
- Tensor 计算时：取 `max_tag_len` 个 Embedding 并 `sum` 或 `mean` 池化

---

### `QuantileEmbSetting` — [`schema/categorical.py`](../../src/betterbole/emb/schema/categorical.py)

**连续值分桶离散化**。

```python
QuantileEmbSetting(
    field_name="duration",
    source=FeatureSource.INTERACTION,
    bucket_count=10,
    embedding_dim=16,
)
```

- `fit()` 计算 `bucket_count - 1` 个分位数边界
- `transform()` 用 `polars.cut()` 将连续值映射到 bucket ID

---

### `MinMaxDenseSetting` — [`schema/numerical.py`](../../src/betterbole/emb/schema/numerical.py)

**连续浮点特征 Min-Max 归一化**。

```python
MinMaxDenseSetting(
    field_name="rating",
    source=FeatureSource.INTERACTION,
)
```

- `fit()` 计算训练集 min/max
- `transform()` 做 `(x - min) / (max - min)` 归一化
- Tensor 计算：直接输出 `[B, 1]` 浮点值（无 Embedding 表）

---

### `VectorDenseSetting` — [`schema/numerical.py`](../../src/betterbole/emb/schema/numerical.py)

**定长向量特征**（如预训练向量）。

```python
VectorDenseSetting(
    field_name="video_vec",
    source=FeatureSource.ITEM,
    embedding_dim=64,    # 如果为 None，自动从训练集推断
)
```

- 自动检查训练集中向量的长度一致性
- Tensor 计算：直接输出 `[B, D]`（无 Embedding 表）

---

### `SequenceSetting` — [`schema/sequence.py`](../../src/betterbole/emb/schema/sequence.py)

**历史行为序列特征**。

```python
SequenceSetting(
    field_name="history_items",
    element_setting=item_id_setting,  # 引用底层 SparseEmbSetting
    group=SeqGroupConfig(
        group_name="item_seq",
        seq_len_field_name="item_seq_len",
        max_len=50,
        padding_side="right",
        time_field_name="history_time",  # 可选
    ),
    truncate_mode="tail",
    is_string_format=False,
    separator=",",
)
```

**核心特性**：
- **不自己建词表**，直接复用 `element_setting` 的词表
- 保证 item 主键和 item 序列的编码空间一致
- 自动产生 `seq_len_field_name` 列
- 支持 `truncate_mode="head"`（保留前 N）或 `"tail"`（保留后 N）
- 可选时间序列对齐

---

### `SeqGroupConfig` — [`schema/base.py`](../../src/betterbole/emb/schema/base.py)

```python
SeqGroupConfig(
    group_name="item_seq",       # 组名，用于 OmniEmbLayer 路由
    seq_len_field_name="item_seq_len",  # 序列长度字段
    max_len=50,                  # 最大序列长度
    padding_side="right",        # 补齐方向
    time_field_name=None,        # 可选时间字段
)
```

---

## 二、Manager 层 — 规则编排

### `SchemaManager` — [`manager.py`](../../src/betterbole/emb/manager.py)

特征工程的总调度引擎。

```python
manager = SchemaManager(
    settings_list=[...],    # List[EmbSetting]
    work_dir="./outputs",
    time_field="timestamp",
    label_fields="label",
    domain_fields="domain",
)
```

#### 自动识别的字段
| 属性 | 来源 |
|------|------|
| `manager.uid_field` | `FeatureSource.USER_ID` 的 setting |
| `manager.iid_field` | `FeatureSource.ITEM_ID` 的 setting |

#### 核心方法

| 方法 | 说明 |
|------|------|
| `fit(train_lf)` | **拟合**：扫描训练集，构建词表/边界 |
| `transform(raw_lf)` | **变换**：应用固化规则到任意 Split |
| `prepare_data(lf)` | 全自动模式：fit + transform 一步完成 |
| `split_dataset(lf, strategy)` | 切分数据集（4种策略） |
| `save_as_dataset(train/valid/test)` | 直接保存已切分数据 |
| `save_schema()` | 固化 `feature_meta.json` |
| `load_schema()` | 从 `feature_meta.json` 恢复 |
| `fields()` | 返回需要保留的字段列表 |
| `make_checkpoint(lf)` | 中间态落盘 |
| `generate_profiles(lf)` | 提取 User/Item 静态画像表 |
| `source2emb_dim(*sources)` | 按 FeatureSource 计算 Embedding 总维度 |

#### 三种运行模式

**方式一：`prepare_data()` 全自动**
```python
manager.prepare_data(lf)  # fit + transform 一把梭
```

**方式二：`fit()` → `transform()` 标准流程**
```python
manager.fit(train_lf)           # 仅在训练集拟合
train = manager.transform(train_lf)
valid = manager.transform(valid_lf)  # 统一词表/OOV
```

**方式三：`fit()` → `split_dataset()` 先拟合再切分**
```python
manager.fit(whole_lf)
manager.split_dataset(whole_lf, strategy="random_ratio")
```

#### 切分策略 — [`split.py`](../../src/betterbole/emb/split.py)

| 策略 | 配置类 | 说明 |
|------|--------|------|
| `loo` | `LooConfig(k_core=3)` | Leave-One-Out，每个用户最晚一行为 test，倒数第二为 valid |
| `time` | `TimeSplitConfig(valid_start, test_start)` | 绝对时间阈值切分 |
| `sequential_ratio` | `SequentialRatioConfig(0.8, 0.1)` | 按时序比例切分 |
| `random_ratio` | `RandomRatioConfig(0.8, 0.1, group_by)` | 随机比例切分，可分组 |

---

## 三、Embedding 层 — 前向计算

### `OmniEmbLayer` — [`emblayer.py`](../../src/betterbole/emb/emblayer.py)

统一 Embedding 层，管理所有字段的 Embedding 查表操作。

```python
omni = OmniEmbLayer(manager=manager)
# 或直接传入 settings
omni = OmniEmbLayer(emb_settings=settings)
```

**预置的 `EmbView` 快捷入口**：

| 视图 | 包含字段 |
|------|---------|
| `omni.whole` | 所有字段 |
| `omni.user_all` | USER_ID + USER |
| `omni.item_all` | ITEM_ID + ITEM |
| `omni.inter` | INTERACTION |
| `omni.user_id` | 仅 USER_ID |
| `omni.item_id` | 仅 ITEM_ID |
| `omni.domain` | domain_fields |
| `omni.domain_id` | 仅 domain_field |

**`EmbView` 使用**：

```python
# 获取所有特征的拼接 Embedding
emb = omni.whole(interaction)  # → Tensor[B, total_dim]

# 按 Source 分组
embs = omni.whole(interaction, split_by="source")
# → {"user_id": Tensor[B, D], "item_id": Tensor[B, D], ...}

# 自定义过滤
emb = omni.whole(interaction, include_fields=["user_id", "item_id"])
emb = omni.whole(interaction, exclude_fields=["domain_id"])
```

**`SeqGroupView`** — 专为序列组设计：

```python
# OmniEmbLayer 自动收集标注了 group_name 的序列特征
seq_group = omni.seq_groups["item_seq"]

# 获取序列 + 目标 + 长度
seq_emb, target_emb, seq_len = seq_group.fetch_all(interaction)
# 带时间
seq_emb, target_emb, seq_len, time = seq_group.fetch_all(interaction, include_time=True)
```

### `RecEmbedding` — [`emblayer.py`](../../src/betterbole/emb/emblayer.py)

对 `nn.Embedding` 的封装，支持自定义初始化：

```python
RecEmbedding(
    num_embeddings=10000,
    embedding_dim=16,
    padding_idx=0,
    init_method='normal',
    init_std=1.0,
)
```

### `ProfileEncoder` — [`emblayer.py`](../../src/betterbole/emb/emblayer.py)

全自动静态画像编码器（已废弃但仍有参考价值）：

- 读取 User/Item Profile Parquet
- 构建 `[max_id, D]` 的静态查询表（`register_buffer`）
- 前向时通过 `query_ids` 查表并走 Embedding 层
