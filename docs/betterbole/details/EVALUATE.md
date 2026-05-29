# 📈 评估模块 (evaluate/)

> **层级**: L3 (评估层)
>
> 依赖 L1 (自包含)。被 L5 (训练层) 的 `BaseTrainer` 调用。

## 架构

```
EvaluatorManager (注册多个评估器)
    │
    ├── Evaluator (统一入口: PointWise + TopK)
    │     ├── PointWiseEvaluator → AUC / LogLoss / GAUC
    │     └── TopKEvaluator      → HR@K / NDCG@K
    │
    └── 每个评估器绑定了 FilterFn（用于分 Domain 评估）
```

---

## `Evaluator` — [`evaluator.py`](../../src/betterbole/evaluate/evaluator.py)

统一的评估入口，自动分流 PointWise 和 TopK 指标。

```python
evaluator = Evaluator(
    "auc", "logloss",          # PointWise 指标
    "hit@10", "ndcg@20",       # TopK 指标
    history_dict=history_dict,  # TopK 必须传历史黑名单
)

# 训练过程中收集
evaluator.collect(
    batch_users,              # Tensor[B]
    batch_targets,            # Tensor[B] (标签)
    batch_preds_1d=scores,    # Tensor[B] (CTR 预测值)
    batch_scores_2d=logits,   # Tensor[B, num_items] (排序分数)
)

# Epoch 结束汇总
results = evaluator.summary()  # {"auc": 0.73, "logloss": 0.41, "hit@10": 0.85, ...}
evaluator.clear()
```

### `PointWiseEvaluator`

用于 CTR 任务的指标计算：

| 指标 | 说明 | 代码位置 |
|------|------|---------|
| `auc` | ROC-AUC | [`metrics.py`](../../src/betterbole/evaluate/metrics.py) `AUC` |
| `logloss` | 交叉熵损失 | [`metrics.py`](../../src/betterbole/evaluate/metrics.py) `LogLoss` |
| `gauc` | 用户分组 AUC (加权) | [`metrics.py`](../../src/betterbole/evaluate/metrics.py) `GAUC` |

### `TopKEvaluator`

用于召回/排序任务的指标计算：

| 指标 | 说明 | 代码位置 |
|------|------|---------|
| `hr@k` / `hit@k` | Hit Rate@K | [`metrics.py`](../../src/betterbole/evaluate/metrics.py) `HR` |
| `ndcg@k` | NDCG@K (归一化折损累计增益) | [`metrics.py`](../../src/betterbole/evaluate/metrics.py) `NDCG` |

**流程**：
1. 对全量物品的预测分数排序
2. Mask 掉用户历史交互（通过 `history_dict`）
3. 取 Top-K，计算命中矩阵
4. 累加每 Batch 的指标值，最后除以总用户数

---

## `EvaluatorManager` — [`manager.py`](../../src/betterbole/evaluate/manager.py)

多评估器管理器，支持**分 Domain** 评估。

```python
manager = EvaluatorManager(
    log_path="./logs/eval.log",
    title="my_experiment",
)

# 注册评估器
manager.register(
    name="overall",             # 全局评估
    evaluator=Evaluator("auc", "logloss"),
)

manager.register(
    name="domain_0",            # 仅评估 domain=0 的样本
    evaluator=Evaluator("gauc"),
    filter_fn=DomainFilter(field_name="domain_id", domain_id=0),
)

# 收集
manager.collect(
    batch_users=batch_users,
    batch_targets=batch_targets,
    batch_inter=batch_interaction,   # 用于 filter_fn 计算 mask
    batch_preds_1d=batch_preds,
)

# 汇总
summary = manager.summary(epoch=epoch)
# {
#   "overall": {"auc": 0.73, "logloss": 0.41},
#   "domain_0": {"gauc": 0.69},
# }
```

### `DomainFilter` — [`manager.py`](../../src/betterbole/evaluate/manager.py)

```python
DomainFilter(field_name="domain_id", domain_id=1)
# → 只放行 domain_id == 1 的样本进入评估器
```

---

## 装饰器

### `LogDecorator` — [`evaluator.py`](../../src/betterbole/evaluate/evaluator.py)

自动将评估结果写入纯文本日志文件。

```python
log_evaluator = LogDecorator(
    evaluator=evaluator,
    save_path="./logs/results.txt",
    title="exp1",
)
results = log_evaluator.summary(epoch=1, step=500)
```

### `EvaluatorDecorator` — [`evaluator.py`](../../src/betterbole/evaluate/evaluator.py)

基类装饰器，可扩展自定义行为。

---

## 指标注册表 — [`metrics.py`](../../src/betterbole/evaluate/metrics.py)

```python
METRIC_REGISTRY = {
    "auc": AUC,
    "logloss": LogLoss,
    "gauc": GAUC,
    "hr": HR,
    "hit": HR,       # hr 和 hit 共享一个实现
    "ndcg": NDCG,
}
```

要添加自定义指标，只需：
1. 继承 `BaseMetric` 并实现 `calculate()` 静态方法
2. 注册到 `METRIC_REGISTRY`
