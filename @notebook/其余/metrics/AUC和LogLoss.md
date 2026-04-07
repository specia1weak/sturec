## Point-wise预测指标
模型会输出CTR类似的绝对概率，指标直接评估这个概率的值吐出得好不好

## sklearn调用
只需要预测和标签就行了，看上去用法很像BCELoss
```python
y_true = np.array([1, 0, 0, 1, 1, 0, 0, 1]) # gt
y_pred = np.array([0.9, 0.1, 0.4, 0.8, 0.65, 0.2, 0.3, 0.7]) # pred

sklearn_logloss = log_loss(y_true, y_pred)
sklearn_auc = roc_auc_score(y_true, y_pred)
```

## 底层计算机制
### 1. logloss: 完全等同BCDLoss值
```python
def calculate_logloss_scratch(y_true, y_pred):
    """
    LogLoss 底层计算：
    公式: - 1/N * sum(y * log(p) + (1-y) * log(1-p))
    注意：为了防止 log(0) 导致数学错误（NaN），通常会对预测值截断极小值 epsilon。
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps) # 限制范围在 [eps, 1-eps] 之间
    loss_sum = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return -np.mean(loss_sum)

```

### 2. AUC
```python
def calculate_auc_scratch(y_true, y_pred):
    """
    AUC 底层计算 (基于排序秩的 Wilcoxon-Mann-Whitney 统计量方法)：
    公式: (正样本的秩和 - M*(M+1)/2) / (M * N)
    """
    # 1. 提取正负样本数量
    M = sum(y_true == 1) # 正样本数
    N = sum(y_true == 0) # 负样本数
    
    # 2. 将概率从小到大排序，获取对应的索引
    # np.argsort 返回的是排序后的索引
    sorted_indices = np.argsort(y_pred)
    
    # 3. 计算正样本的 Rank（秩/排名）之和
    # 排名从 1 开始，所以需要索引 + 1
    rank_sum = 0
    for rank, idx in enumerate(sorted_indices, start=1):
        if y_true[idx] == 1:
            rank_sum += rank
            
    # 注：真实的工业界代码通常会处理概率完全相同（tie）的情况，赋予平均排名。
    # 这里为了展示核心逻辑，假设概率不重复。
    
    # 4. 套用公式计算面积
    auc = (rank_sum - M * (M + 1) / 2) / (M * N)
    return auc
```

### 3. GAUC
比如，用户 A 是重度依赖者，真实点击率 20%，模型给他的物品打分普遍在 0.5 左右；用户 B 是新用户，真实点击率 1%，模型给他的物品打分普遍在 0.05 左右。
如果在全局算 AUC，模型只要把 A 的所有物品排在 B 前面就能拿高分，但这并不代表模型能给 A 或 B 挑出他们各自最喜欢的物品。