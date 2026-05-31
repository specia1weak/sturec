# evaluate/

这一层负责把模型输出变成可读指标。

## 1. 真实的指标实现

源码在 [`src/betterbole/evaluate/metrics.py`](../../src/betterbole/evaluate/metrics.py)。

### `AUC`

- 输入：`y_true`, `y_pred`
- 实现：`sklearn.metrics.roc_auc_score`
- 如果标签只有单一类别，会返回 `0.0`

### `LogLoss`

- 输入：`y_true`, `y_pred`
- 实现：`sklearn.metrics.log_loss`
- 这里要求 `y_pred` 是概率值，不是 raw logits

### `GAUC`

- 输入：`y_true`, `y_pred`, `users`
- 先按 user 分组，再做加权 AUC

### `HR`

- 输入：`hits_k`
- 返回命中数总和

### `NDCG`

- 输入：`hits_k`, `k`
- 返回按折损权重累计的得分

### `METRIC_REGISTRY`

当前注册表实际包含：

```python
{
    "auc": AUC,
    "logloss": LogLoss,
    "gauc": GAUC,
    "hr": HR,
    "ndcg": NDCG,
}
```

## 2. `BaseEvaluator`

源码在 [`src/betterbole/evaluate/evaluator.py`](../../src/betterbole/evaluate/evaluator.py)。

`BaseEvaluator` 只是一个抽象壳，真正干活的是 `PointWiseEvaluator` 和 `TopKEvaluator`。

## 3. `PointWiseEvaluator`

```python
evaluator = PointWiseEvaluator(["auc", "logloss"])
evaluator.collect(batch_users, batch_targets, batch_preds)
results = evaluator.summary()
```

它会把所有 batch 的 user / target / pred 先攒起来，最后统一算指标。

## 4. `TopKEvaluator`

```python
evaluator = TopKEvaluator(["hr@10", "ndcg@20"], history_dict=history_dict)
evaluator.collect(batch_users, batch_targets, batch_scores)
```

### 关键点

- `batch_scores` 必须是 `[B, num_items]`
- `history_dict` 用来把用户历史交互从候选里 mask 掉
- `TopKEvaluator` 里会先把 target 位置的分数还原，再做 top-k

### 当前代码和文档要对齐的地方

- `VALID_METRICS` 里允许 `recall`，但 `METRIC_REGISTRY` 没有 `recall` 实现，所以 `recall@k` 现在不能当成稳定可用指标写进主文档。
- `hit@k` 不是当前代码支持的标准名字，当前稳定写法是 `hr@k`。

## 5. `Evaluator`

这是最常用的入口。

```python
evaluator = Evaluator("auc", "logloss", "hr@10", history_dict=history_dict)
```

它会自动分流：

- 不带 `@` 的指标走 `PointWiseEvaluator`
- 带 `@` 的指标走 `TopKEvaluator`

### 收集

```python
evaluator.collect(
    batch_users=batch_users,
    batch_targets=batch_targets,
    batch_preds_1d=batch_preds,
    batch_scores_2d=batch_scores,
)
```

### 汇总

```python
results = evaluator.summary()
evaluator.clear()
```

## 6. `EvaluatorManager`

源码在 [`src/betterbole/evaluate/manager.py`](../../src/betterbole/evaluate/manager.py)。

它的作用是“多个 evaluator 并行跑”，并且可以按 domain 过滤。

### 注册

```python
manager = EvaluatorManager(log_path="./logs/eval.log", title="demo")
manager.register("overall", Evaluator("auc", "logloss"))
manager.register(
    "domain_0",
    Evaluator("auc"),
    filter_fn=DomainFilter(field_name="domain_id", domain_id=0),
)
```

### 收集

```python
manager.collect(
    batch_users=batch_users,
    batch_targets=batch_targets,
    batch_inter=batch_interaction,
    batch_preds_1d=batch_preds,
)
```

### 汇总

```python
summary = manager.summary(epoch=1, step=100)
```

返回结构是：

```python
{
    "overall": {"auc": 0.73, "logloss": 0.41},
    "domain_0": {"auc": 0.69},
}
```

## 7. `LogDecorator`

它是 `Evaluator` 的日志外壳，会把结果写到文本文件。

## 8. 实战建议

- 只看 AUC 时，`predict()` 返回 raw logits 也能接受。
- 要算 `logloss`，建议让 `predict()` 输出概率值，或者自己在评估前做 sigmoid。
- `EvaluatorManager` 的 `filter_fn` 只负责返回 mask，不负责修改数据。
