# KuaiRand1K AUC Gap 分析报告
## studybole (0.72) vs Crocodile (0.785)

---

## 一、问题背景

同样是 KuaiRand1K 数据集，同样是 CTR 预测任务：
- **studybole (kuairan1k.py)**：PLE 模型，AUC ≈ 0.72
- **Crocodile (论文源码)**：SharedBottom 模型，AUC ≈ 0.785

一个更简单的模型跑出更高的指标，说明差距来源于**数据处理与训练流程**，而非模型复杂度本身。

以下按影响从大到小排列所有关键差异。

---

## 二、关键差异分析

### 差异 1：稀有类别处理（min_categr_count）与 padding_idx 机制【最高影响】

#### 1a. Crocodile 的完整实现流程

**第一步：Tokenizer 构建词表（`preprocess/utils.py`）**

```python
class Tokenizer:
    def build_vocab(self, word_counts):
        words = []
        for token, count in word_counts.items():
            if count >= self._min_freq:   # ← 核心：频次过滤
                words.append(token)
        words.sort()
        # 词表布局：
        self.vocab["__PAD__"] = 0          # idx 0：专用 padding（全零，不训练）
        # idx 1 .. N：频次 >= min_freq 的合法 token，每个有独立 embedding
        self.vocab["__OOV__"] = vocab_size # idx N+1：所有稀有/未见 token 共享此 embedding
```

所以词表的 index 布局是：

| idx | 含义 |
|-----|------|
| 0 | `__PAD__`（序列补位/缺失值，embedding 全零，**不参与训练**） |
| 1 ~ N | 训练集中出现次数 ≥ `min_freq` 的合法类别，各有独立 embedding |
| N+1 | `__OOV__`（出现次数 < `min_freq` 的稀有类别，**全部共享**此 embedding） |

**第二步：Embedding 创建（`layers/embeddings/feature_embedding.py`）**

```python
padding_idx = feature_spec.get("padding_idx", None)  # 从 feature_map 读到 0
embedding_matrix = nn.Embedding(vocab_size, feat_emb_dim, padding_idx=padding_idx)
```

PyTorch `nn.Embedding(padding_idx=0)` 的效果：
- idx 0 的 weight 行被强制固定为全零向量
- 梯度不会更新 idx 0（即使有样本落在上面也无效）

**第三步：参数初始化（同文件 `reset_parameters`）**

```python
if v.padding_idx is not None:
    self.embedding_initializer(v.weight[1:, :])  # 只初始化 idx 1..N+1
    # idx 0 保持全零（PyTorch 保证）
```

**第四步：编码时的查询（`fit_categorical_col`）**

```python
# feature_map 中记录：
"padding_idx": 0,        # PAD at idx 0
"oov_idx": N+1,          # OOV at last index
"vocab_size": N+2        # total = PAD + N_valid + OOV
```

在 `encode_category` 中：
```python
category_indices = [self.vocab.get(x, self.vocab["__OOV__"]) for x in categories]
# 训练集中频次 < 10 的 token → 查不到 → 返回 __OOV__ idx
# 验证/测试集中从未见过的 token → 同样返回 __OOV__ idx
```

**关键结论**：
- `__OOV__` embedding（idx N+1）**是可训练的**，它从所有稀有类别的样本中**聚合梯度**，学到"通用低频 item"的表示
- `__PAD__` embedding（idx 0）**永远是全零，不可训练**

---

#### 1b. studybole 的实现（`schema.py` + `emblayer.py`）

**词表构建（`SparseEmbSetting.parse_fit_result`）**：

```python
unique_vals = sorted(unique_vals)
self.vocab = {str(val): idx + 1 for idx, val in enumerate(unique_vals)}
# idx 布局：0 → 保留给 OOV（通过 replace_strict 的 default=0 实现）
#           1 ~ N → 训练集中所有唯一值，不论出现几次
```

**Transform（`get_transform_expr`）**：

```python
pl.col(field).replace_strict(self.vocab, default=pl.lit(0, dtype=pl.UInt32))
# 训练集中未出现的值（验证集新 token） → idx 0
```

**Embedding 创建（`emblayer.py BoleEmbLayer`）**：

```python
nn.Embedding(
    num_embeddings=setting.num_embeddings,
    embedding_dim=setting.embedding_size,
    padding_idx={True: 0, False: None}[setting.padding_zero]  # 默认 padding_idx=0
)
```

studybole 同样使用了 `padding_idx=0`，idx 0 是全零不训练的。

---

#### 1c. 两者的核心差异对比

| 情况 | Crocodile | studybole |
|------|-----------|-----------|
| 训练集出现次数 ≥ 10 的 token | 独立 idx（可训练） | 独立 idx（可训练）|
| 训练集出现次数 1~9 的**稀有** token | → OOV idx（**可训练，共享聚合梯度**） | → 独立 idx（可训练，但梯度极少，接近随机噪声）|
| 验证/测试集中从未见过的 token | → OOV idx（**可训练，有意义的通用表示**） | → idx 0（**全零向量，无任何表示**） |

**影响分析**：

在 KuaiRand1K 中，`video_id`、`author_id`、`music_id` 都存在大量低频 ID（只在训练集出现 1~9 次）。studybole 的处理导致：
- 这些 ID 的 embedding 几乎没有有效梯度，接近于随机初始化
- 验证集中见到这些 ID 时，模型的预测基本靠噪声
- Embedding 表远比 Crocodile 大（更多参数，更难训练）

Crocodile 的 `__OOV__` embedding 会聚合所有低频 ID 的梯度，学到"低频/冷启动 item 通常的行为模式"，验证时能给出有意义的预测。

**这是最大的 AUC 差距来源。**

---

### 差异 2：场景/Tab 选择逻辑【高影响】

**Crocodile（1_preprocess_MDST.py）**：
```python
selected_scenario = [0, 1, 2, 4, 6]  # 明确指定 5 个场景
```
论文硬编码选择了第 0、1、2、4、6 号 tab，这是论文作者经过研究后手动选择的具有代表性的场景。

**studybole（kuairan1k.py）**：
```python
top_5_domains = (
    whole_lf.group_by("tab")
    .len()
    .sort("len", descending=True)
    .head(5)
    .select("tab")
)
whole_lf = whole_lf.join(top_5_domains, on="tab", how="semi")
```
取**全量数据**中交互次数最多的 5 个 tab。

**问题所在**：
- KuaiRand1K 的 tab 0~9 中，频次最高的 5 个未必是 [0,1,2,4,6]
- 如果 studybole 选入了 tab 3 或 tab 5（被论文排除的场景），这些场景的点击率分布可能截然不同
- 不同的场景组合直接决定了数据的正负样本比例和类间分布难度，从而影响 AUC 上限
- 可能选入了"太容易"或"太难"的场景，导致 AUC 偏离论文设置

**验证方法**：打印 `top_5_domains.collect()`，看选出的 tab 是否与 [0,1,2,4,6] 相同。

---

### 差异 3：验证集划分范围【高影响】

**Crocodile（0_preprocess.py）**：
```python
train_data = data[data['date'] <= 20220506]
valid_data = data[data['date'] == 20220507]   # 仅1天
test_data  = data[data['date'] == 20220508]   # 仅1天
```

**studybole（kuairan1k.py）**：
```python
train_lf = transformed_lf.filter(pl.col("date") <= 20220506)
valid_lf = transformed_lf.filter(pl.col("date") >= 20220507)  # 2天：20220507 + 20220508
```

**影响**：
- studybole 把 Crocodile 的 **test 集（20220508）也混入了验证集**
- 这改变了验证集的数据分布（两天行为混合）
- 导致验证集评估的 AUC 可能偏低（两天数据的用户行为时间偏移会引入噪声）
- 更重要的是：**论文报告的是 20220507 这一天的 AUC，studybole 报告的是两天合并的 AUC**，两个数值本就不具备直接可比性

---

### 差异 4：无 Early Stopping 和最优模型保存【高影响】

**Crocodile（rank_model.py + model_config.yaml）**：
```yaml
epochs: 50
early_stop_patience: 2
monitor: 'AUC'
monitor_mode: 'max'
save_best_only: True
reduce_lr_on_plateau: True
```
- 监控验证集 AUC，patience=2 则停止
- **自动保存最高 AUC 对应的 checkpoint**
- 训练结束后 load 最好的模型，再报告该模型的性能
- 学习率在 plateau 时衰减（× 0.1）

**studybole（kuairan1k.py）**：
```python
for epoch in range(20):
    ...  # 训练
    metrics_result = evaluator.summary(epoch)
    print(f"Validation Metrics: {metrics_result}")
    evaluator.clear()
```
- 固定训练 20 个 epoch
- **没有 early stopping**
- **没有最优 checkpoint 保存**
- 报告的是**最后一个 epoch** 的验证集 AUC

**影响**：
- 如果模型在 epoch 8~10 达到最优，之后开始轻微过拟合，Crocodile 会报告峰值，studybole 会报告最后一个 epoch 的较低值
- 没有 LR 衰减，模型在后期无法精细收敛

---

### 差异 5：特征集合差异【中等影响】

| 特征 | Crocodile | studybole |
|------|-----------|-----------|
| `onehot_feat0` | ❌ 未使用 | ✅ 有 |
| `visible_status` | ❌ 未使用 | ✅ 有 |
| `server_width/height` | ❌ 未使用 | ✅ 有 |
| `day_of_week` | ❌ 未使用 | ✅ 有（工程特征）|
| `hour` | ❌ 未使用 | ✅ 有（工程特征）|
| `is_weekend` | ❌ 未使用 | ✅ 有（工程特征）|

**分析**：
- studybole 加入了额外的工程特征（时间相关）和额外的 item 特征，**理论上应该有帮助**
- 但如果同时没有 min_count 过滤（稀有 embedding 噪声大），这些特征的信号被稀有 ID 的噪声淹没
- `server_width/height` 可能对 CTR 预测价值不大（视频服务器分辨率与用户点击关联性弱）
- Crocodile 不含 `onehot_feat0`（可能是有意排除，避免信息泄露或冗余）

---

### 差异 6：Tag 特征处理方式【中等影响】

**Crocodile（dataset_config.yaml）**：
```yaml
{ name: 'tag', active: True, dtype: str, splitter: ^, type: sequence, max_len: 5, padding: pre }
```
- Tag 被当做**有序序列**，最多取前 5 个 tag
- 使用序列 embedding 并 pooling

**studybole（kuairan1k.py）**：
```python
SparseSetEmbSetting("tag", FeatureSource.ITEM, 16, is_string_format=True, separator=",")
```
- Tag 被当做**无序集合**，所有 tag 的 embedding 平均
- 没有 max_len 限制

**影响**：
- 两种方式差距不大，但 Crocodile 限定 max_len=5 可以减少噪声（避免包含太多无关 tag）
- 平均池化 vs 序列 pooling 差别不显著

---

### 差异 7：正则化方式【低影响】

**Crocodile**：
```yaml
net_regularizer: 1.e-6
embedding_regularizer: 0.
```
+ `max_gradient_norm=10.`（梯度裁剪）

**studybole**：
```python
optimizer = torch.optim.Adam(named_parameters, lr=1e-3)
# weight_decay=1e-5 仅对非 embedding 参数
```
- 无梯度裁剪
- embedding 无正则化（两者一致）
- 网络正则化系数有差异（1e-6 vs 1e-5），差异较小

---

## 三、差距量化估计

| 原因 | 估计 AUC 影响 |
|------|--------------|
| min_categr_count=10 缺失（稀有 ID 噪声） | -0.02 ~ -0.04 |
| Tab 选择不同（可能混入不同场景） | -0.01 ~ -0.03 |
| 无 Early Stopping（报告最后 epoch 而非最优） | -0.01 ~ -0.02 |
| 验证集包含两天 vs 一天 | -0.005 ~ -0.01 |
| 其余特征/正则化差异 | <0.005 |

---

## 四、建议的修复方案

### 修复 1（最高优先）：在 SparseEmbSetting 中实现 min_freq + OOV 机制

需要让稀有 token 映射到一个**可训练的** OOV embedding（末尾 idx），而不是映射到 idx 0（全零不可训练）。这与 Crocodile 的行为完全对齐。

**修改 `schema.py` 的 `SparseEmbSetting`**：

```python
class SparseEmbSetting(EmbSetting):
    def __init__(self, field_name, source, embedding_size=16, num_embeddings=-1,
                 padding_zero=True, min_freq=1):   # ← 新增 min_freq
        super().__init__(field_name, embedding_size, source, padding_zero)
        self._num_embeddings = num_embeddings
        self.vocab: Dict[str, int] = {}
        self.min_freq = min_freq                   # ← 记录最低频次
        self.oov_idx = -1                          # ← 记录 OOV 所在 idx
        if num_embeddings > 0:
            self.is_fitted = True

    def get_fit_exprs(self):
        # 改为返回频次统计，而不只是 unique 值
        return [
            pl.col(self.field_name).cast(pl.Utf8).drop_nulls()
            .value_counts()                        # ← 统计每个值出现次数
            .implode()
            .alias(self.field_name)
        ]

    def parse_fit_result(self, result_df):
        rows = result_df.get_column(self.field_name).to_list()[0]
        # rows 是 [{"value": "xxx", "count": N}, ...] 的列表
        valid_vals = sorted([
            r["value"] for r in rows
            if r["value"] is not None and r["count"] >= self.min_freq  # ← 频次过滤
        ])
        if self.padding_zero:
            # idx 0：PAD（全零，不训练）
            # idx 1..N：合法 token
            # idx N+1：OOV（稀有/未见 token，可训练）
            self.vocab = {str(val): idx + 1 for idx, val in enumerate(valid_vals)}
            self.oov_idx = len(self.vocab) + 1
            self._num_embeddings = self.oov_idx + 1   # PAD + N_valid + OOV
        else:
            self.vocab = {str(val): idx for idx, val in enumerate(valid_vals)}
            self.oov_idx = len(self.vocab)
            self._num_embeddings = self.oov_idx + 1
        self.is_fitted = True

    def get_transform_expr(self):
        return (
            pl.col(self.field_name)
            .cast(pl.Utf8)
            .replace_strict(self.vocab, default=pl.lit(self.oov_idx, dtype=pl.UInt32))  # ← 稀有/未见 → oov_idx
            .cast(pl.UInt32)
            .alias(self.field_name)
        )
```

这样：
- 训练集中频次 ≥ `min_freq` 的 token → 独立 idx（1..N），可训练
- 频次 < `min_freq` 的稀有 token 以及验证集新 token → `oov_idx`（N+1），**可训练，共享梯度**
- idx 0 依然是全零 PAD，不参与训练

**在 kuairan1k.py 中使用**：

```python
# 对高基数 ID 类特征加 min_freq=10
user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, 32, min_freq=10)
item_setting = SparseEmbSetting("video_id", FeatureSource.ITEM_ID, 32, min_freq=10)
# item 属性特征也需要
SparseEmbSetting("author_id", FeatureSource.ITEM, 32, min_freq=10),
SparseEmbSetting("music_id", FeatureSource.ITEM, 32, min_freq=10),
# 低基数特征（类别少，几乎没有稀有值）可以保持 min_freq=1
SparseEmbSetting("tab", FeatureSource.INTERACTION, 16, min_freq=1),
```

### 修复 2：固定 Tab 选择为 [0, 1, 2, 4, 6]

```python
# 替换动态 top-5 选择：
whole_lf = whole_lf.filter(pl.col("tab").is_in([0, 1, 2, 4, 6]))
```

### 修复 3：修正验证集划分

```python
train_lf = transformed_lf.filter(pl.col("date") <= 20220506)
valid_lf = transformed_lf.filter(pl.col("date") == 20220507)  # 仅用 20220507
test_lf  = transformed_lf.filter(pl.col("date") == 20220508)  # 20220508 作 test
```

### 修复 4：加入 Early Stopping 和最优 Checkpoint

```python
best_auc = 0.0
best_epoch = 0
for epoch in range(50):  # 扩大 epoch 上限，依赖 early stopping
    # 训练...
    metrics_result = evaluator.summary(epoch)
    current_auc = metrics_result.get('auc', 0.0)
    
    if current_auc > best_auc:
        best_auc = current_auc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 2:
            print(f"Early stopping at epoch {epoch}, best AUC={best_auc:.4f} at epoch {best_epoch}")
            break

# 加载最优模型
model.load_state_dict(torch.load('best_model.pt'))
```

---

## 五、结论

0.72 vs 0.785 的差距（约 0.065 AUC）主要来源于：

1. **稀有 ID 没有 OOV 机制** → embedding 噪声大，泛化能力差（最主要原因）
2. **场景/Tab 选择不同** → 两个实验评测的实际是不同的数据子集
3. **没有 Early Stopping** → 报告的是最后一个 epoch 而非最优 epoch 的结果
4. **验证集包含 2 天** → 与论文的单天验证不可直接比较

这些都是工程实现层面的问题，与 PLE 模型架构本身无关。按照上述修复方案调整后，studybole 的 PLE 应当可以达到与 Crocodile SharedBottom 持平甚至更高的 AUC。