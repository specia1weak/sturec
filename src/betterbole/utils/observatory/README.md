# Observatory

`observatory` 是 BetterBole 的训练时表示监控包。它的定位不是“自动帮你做完所有分析”，而是提供一套足够轻量、足够统一的接口，让模型作者可以在训练过程中持续回答下面这些问题：

- 某个分支是不是已经塌缩了。
- gate 是不是真的在路由，还是只是软平均。
- shared / specific / context / innovation 之间有没有学出分工。
- 哪些维度最活跃。
- 这些“活跃”到底是独立自由度，还是只是少数主方向的重复拷贝。

如果你现在的感觉是“`TensorObservatory` 看不懂怎么用”，可以直接跳到本文的这几个部分：

- `3. 最小可用例子`
- `4. 训练里怎么接`
- `5. 怎么读输出`
- `6. 怎么导出 history / heatmap / plot`

---

## 1. 先记住它是什么

`TensorObservatory` 本质上是一个“按名字缓存张量窗口统计”的对象。

它做的事情很简单：

- 你给它一个名字和一个 tensor。
- 它把 tensor 拉平、转到 CPU、做基础统计。
- 它只保留最近 `window_size` 次记录。
- 需要时，你再把这些窗口统计渲染成文本、history、step x dim 矩阵，或者自己画图。

它默认不会：

- 自动打印所有内容。
- 自动写文件。
- 自动在训练里接线。
- 自动知道你想监控哪些张量。

也就是说，它是一个“分析缓存层 + 文本输出层”，不是一个全自动实验系统。

---

## 2. 核心对象一览

最常用的对象只有 4 个：

- `TensorObservatory`
  - 主对象，负责记录张量和聚合窗口统计。
- `TensorMonitorOptions`
  - 控制某个 tensor 要算什么指标、显示多少维度、sketch 保留多大。
- `TensorDisplayConfig`
  - 控制终端展示格式。
- `RelationOptions`
  - 控制多 tensor 之间的关系分析。

从包里导入通常这么写：

```python
from betterbole.utils.observatory import (
    RelationOptions,
    TensorDisplayConfig,
    TensorMonitorOptions,
    TensorObservatory,
)
```

如果你要自己做谱分析，也可以直接用：

```python
from betterbole.utils.observatory.metrics import compute_spectrum
```

---

## 3. 最小可用例子

下面这个例子已经覆盖了 80% 的基本用法。

```python
import torch

from betterbole.utils.observatory import (
    RelationOptions,
    TensorDisplayConfig,
    TensorMonitorOptions,
    TensorObservatory,
)


obs = TensorObservatory(
    window_size=20,
    relation_options=RelationOptions(
        enabled=True,
        rank=8,
        max_pairs=8,
        names=("shared_hidden", "specific_hidden"),
    ),
)

vector_options = TensorMonitorOptions(
    display=TensorDisplayConfig(
        show_global_summary=True,
        show_per_dim=True,
        max_display_dims=12,
        topk_display_dims=8,
        rank_by="variance",
    )
)

obs.register("shared_hidden", vector_options)
obs.register("specific_hidden", vector_options)

for step in range(1, 6):
    shared = torch.randn(1024, 64)
    specific = 0.3 * torch.randn(1024, 64) + shared[:, :1]

    obs.record("shared_hidden", shared, step=step)
    obs.record("specific_hidden", specific, step=step)

print(
    obs.get_window_stats(
        names=["shared_hidden", "specific_hidden"],
        include_relations=True,
    )
)
```

你会得到：

- 每个 tensor 的全局摘要
- top-k 活跃维度表
- `shared_hidden` 和 `specific_hidden` 的关系分析

---

## 4. 训练里怎么接

最常见的接法是下面这种。

### 4.1 在模型初始化或 setup 阶段注册

```python
self.observatory = TensorObservatory(window_size=50)

feature_options = TensorMonitorOptions(
    display=TensorDisplayConfig(
        show_global_summary=True,
        show_per_dim=True,
        max_display_dims=16,
        topk_display_dims=8,
        rank_by="variance",
    )
)

self.observatory.register("shared_hidden", feature_options)
self.observatory.register("specific_hidden", feature_options)
self.observatory.register("gate_weights", feature_options)
```

为什么建议先 `register`：

- 可以预先给每个名字绑定配置。
- 方便统一管理监控对象。
- 不容易在训练中途拼错名字。

虽然 `record()` 会自动补注册，但自动补注册时只会用默认配置。

### 4.2 在 forward 或 train step 里 record

```python
def custom_train_step(self, batch_interaction, ctx):
    loss = self.calculate_loss(batch_interaction)
    loss.backward()
    ctx.optimizer.step()

    step = ctx.global_step + 1
    self.observatory.record("shared_hidden", self._latest_debug["shared_hidden"], step=step)
    self.observatory.record("specific_hidden", self._latest_debug["specific_hidden"], step=step)
    self.observatory.record("gate_weights", self._latest_debug["gate_weights"], step=step)

    if step % 200 == 0:
        print(
            self.observatory.get_window_stats(
                names=["shared_hidden", "specific_hidden", "gate_weights"],
                include_relations=True,
                relation_names=["shared_hidden", "specific_hidden"],
            )
        )

    return float(loss.item())
```

### 4.3 在 epoch end 导出历史和图

`TensorObservatory` 自己不画图，但它能把 history 和 step x dim 矩阵吐给 plotting 层。

比如：

```python
steps, values = self.observatory.get_scalar_history("shared_hidden", "effective_rank")
steps2, matrix = self.observatory.get_step_dim_matrix("shared_hidden", "batch_var")
```

然后你可以交给：

- `plot_multi_series`
- `plot_ranked_profile`
- `plot_step_dim_heatmap`
- `plot_topk_bar`

---

## 5. 它到底记录了什么

一次 `record(name, tensor, step=...)` 会做下面这些事情：

1. `tensor.detach().float().cpu()`
2. 拉平成二维
3. 计算该次 batch 的基础统计
4. 截取轻量 sketch
5. 只把统计结果和 sketch 放进窗口缓存

### 5.1 拉平规则

拉平规则非常重要，因为很多指标都是在拉平后的矩阵上算的。

- 标量 `[]`
  - 变成 `(1, 1)`
- 向量 `[N]`
  - 变成 `(N, 1)`
- 张量 `[B, ...]`
  - 变成 `(B, -1)`

也就是说：

- 如果你记录 `(batch, hidden_dim)` 的表征，完全符合预期。
- 如果你记录 `(batch, seq_len, hidden_dim)`，它会被看成 `(batch, seq_len * hidden_dim)`。

相关代码在 [analysis.py](D:\pyprojects\recommend-study\studybole\src\betterbole\utils\observatory\analysis.py:6)。

### 5.2 每条记录里保存的字段

每次 `record()` 后，内部会保存一条 entry，里面至少有：

- `step`
- `batch_mean`
- `batch_var`
- `feature_mean`
- `feature_var`
- `num_samples`
- `flat_dim`
- `sketch`

如果配置里启用了 `spectral / correlation / cosine`，还会多算：

- `effective_rank`
- `participation_ratio`
- `stable_rank`
- `top1_energy_ratio`
- `top2_energy_ratio`
- `dead_dim_ratio`
- `mean_dim_var`
- `max_dim_var`
- `mean_abs_corr`
- `max_abs_corr`
- `sample_cosine_mean`
- `sample_cosine_abs_mean`

---

## 6. 最重要的几个 API

### 6.1 `register(name, options=None)`

用途：

- 提前声明一个 tensor 名字。
- 给这个名字绑定专属配置。

例子：

```python
obs.register("shared_hidden", feature_options)
```

### 6.2 `record(name, tensor_data, options=None, step=None)`

用途：

- 记录一次张量快照。

说明：

- `step` 推荐显式传入。
- 不传也能用，但会按名字各自递增计数。
- `options` 如果传入，会覆盖该名字当前配置。

例子：

```python
obs.record("shared_hidden", shared_hidden, step=1200)
```

### 6.3 `get_window_stats(...)`

用途：

- 输出人能直接读的文本报告。

例子：

```python
text = obs.get_window_stats(
    names=["shared_hidden", "specific_hidden"],
    include_relations=True,
    relation_names=["shared_hidden", "specific_hidden"],
)
print(text)
```

它输出的不是“最后一次 raw tensor”，而是最近 `window_size` 条记录的窗口聚合统计。

### 6.4 `get_relation_stats(...)`

用途：

- 只看 tensor 之间的关系，不看单 tensor 统计。

例子：

```python
print(
    obs.get_relation_stats(
        names=["shared_hidden", "specific_hidden", "context_hidden"],
        max_pairs=6,
    )
)
```

### 6.5 `get_scalar_history(name, key)`

用途：

- 拿某个标量指标的历史曲线。

例子：

```python
steps, eff_rank = obs.get_scalar_history("shared_hidden", "effective_rank")
```

适合画：

- `effective_rank`
- `feature_var`
- `top1_energy_ratio`

### 6.6 `get_vector_history(name, key)`

用途：

- 拿某个向量指标在每个 step 的历史。

例子：

```python
steps, batch_var_series = obs.get_vector_history("shared_hidden", "batch_var")
```

适合做：

- top-k dim 方差变化
- 某些维度的随 step 曲线

### 6.7 `get_step_dim_matrix(name, key)`

用途：

- 直接拿到 `step x dim` 矩阵。

例子：

```python
steps, matrix = obs.get_step_dim_matrix("shared_hidden", "batch_var")
```

适合画：

- step x dim heatmap
- 维度坍缩轨迹

---

## 7. 配置项到底怎么选

### 7.1 `TensorDisplayConfig`

最常改的是这几个字段：

- `show_global_summary`
  - 是否显示全局摘要表
- `show_per_dim`
  - 是否显示逐维表
- `max_display_dims`
  - 如果总维度不超过这个值，就把所有维度全打出来
- `topk_display_dims`
  - 如果维度太多，只打 top-k
- `rank_by`
  - top-k 的排序依据
  - 可选值常用是 `variance`、`mean_abs`、`train_var`

典型配置：

```python
TensorDisplayConfig(
    show_global_summary=True,
    show_per_dim=True,
    max_display_dims=12,
    topk_display_dims=8,
    rank_by="variance",
)
```

### 7.2 `TensorMonitorOptions`

它控制两件事：

- 算哪些指标
- sketch 保留多大

默认：

```python
TensorMonitorOptions(
    metrics=("basic", "spectral", "correlation", "cosine")
)
```

如果你只想轻量监控，可以只保留基础项。  
如果你想看秩塌缩和子空间关系，就保留 `spectral / correlation / cosine`。

### 7.3 `TensorSketchConfig`

最常用的是：

- `max_samples`
  - sketch 最多保留多少行
- `max_dims`
  - sketch 最多保留多少维

注意：

- 关系分析、相关性、SVD 等更重的统计，大多是基于 sketch 算的。
- `max_samples` 和 `max_dims` 设太大，会明显拖慢训练中的诊断部分。

---

## 8. `window_size` 到底是什么意思

很多人第一次会误解这一点。

`window_size=50` 的意思不是“每个 tensor 最多保存 50 个维度”，而是：

- 每个 tensor 只保留最近 50 次 `record()` 的 entry

因此：

- `get_window_stats()` 是“最近 50 次记录的聚合”
- 不是“全训练历史”

如果你想看全历史曲线，有两个办法：

- 训练过程中更早导出
- 或者把 `window_size` 设大一些

---

## 9. 关系分析是怎么工作的

关系分析读的是“最近一次记录的 sketch”，不是整个窗口的所有原始张量。

目前关系表里主要有：

- `linear_cka`
  - 两个表示整体相似度
- `subspace_mean_cos`
  - 两个主子空间的平均重合程度
- `subspace_min_cos`
  - 主子空间里最弱的重合程度

经验上可以这么看：

- `linear_cka` 很高
  - 两个分支很可能在学相似东西
- `subspace_mean_cos` 很高
  - 两个分支主方向接近
- `subspace_min_cos` 很低
  - 有些方向并不共享，仍有局部差异

例子：

```python
obs.configure_relations(
    RelationOptions(
        enabled=True,
        rank=8,
        max_pairs=12,
        names=("shared_hidden", "specific_hidden", "context_hidden"),
    )
)
```

---

## 10. 怎么读终端输出

默认终端输出分成两块。

### 10.1 全局摘要表

你会看到这些列：

- `feat_mean`
  - 整体均值
- `feat_var`
  - 整体方差
- `n`
  - 样本数
- `dim`
  - 拉平后的维度数
- `eff_rank`
  - 有效秩
- `part_ratio`
  - participation ratio
- `stable_rank`
  - 稳定秩
- `top1_energy`
  - 第一奇异方向能量占比
- `top2_energy`
  - 前两奇异方向能量占比
- `dead_dim`
  - 死维比例
- `mean_dim_var`
  - 平均逐维方差
- `max_dim_var`
  - 最大逐维方差
- `mean_abs_corr`
  - 平均绝对相关性
- `max_abs_corr`
  - 最大绝对相关性
- `cos_mean`
  - 样本对之间平均余弦
- `abs_cos_mean`
  - 样本对之间平均余弦绝对值

### 10.2 `[Top-Dim]` 或 `[Per-Dim]`

这一段显示逐维统计：

- `dim_idx`
- `batch_mean`
- `batch_var`
- `train_var`

含义：

- `batch_var`
  - 当前窗口里，单维在 batch 内的平均方差
- `train_var`
  - 当前窗口里，这一维均值跨 step 的波动

如果维度数很少，会显示 `[Per-Dim]`。  
如果维度很多，只会显示排序后的 top-k，会显示 `[Top-Dim]`。

---

## 11. 奇异值、有效秩、单维方差之间的关系

这是 observatory 最值得用的一块。

### 11.1 奇异值是什么

对中心化后的特征矩阵做 SVD：

```text
X_centered = U Σ V^T
```

`Σ` 对角线上的值就是奇异值。它描述的是：

- 特征变化主要沿哪些方向展开
- 每个主方向有多强

### 11.2 它不等于“只有几个原始维度在动”

这点非常重要。

- 单维方差大
  - 说明某个原始坐标轴活跃
- 奇异值前几项特别大
  - 说明整体变化集中在少数几个线性组合方向

所以一个张量可能：

- 很多维度方差都不小
- 但这些维度彼此高度相关
- 最后 SVD 仍然告诉你，它本质上只有 1 到 2 个自由度

### 11.3 实战怎么判断塌缩

典型低秩塌缩特征：

- `eff_rank` 接近 1 到 3
- `top1_energy` 接近 1
- `mean_abs_corr` 很高

典型健康表征特征：

- `eff_rank` 不低
- `top1_energy` 不高
- `mean_abs_corr` 中低

---

## 12. 常见接入模式

### 模式 A：只把它当文本 recorder

适合：

- 快速看 shared/specific 输出
- 快速看 gate 权重是否均衡

用法：

- `register`
- `record`
- `get_window_stats`

### 模式 B：做 step 曲线分析

适合：

- 看 `effective_rank` 是否逐步塌缩
- 看 `feature_var` 是否训练后期归零

用法：

- `record(..., step=...)`
- `get_scalar_history`

### 模式 C：做维度热图分析

适合：

- 看某几维是不是一直最活跃
- 看维度表达是否逐步死亡

用法：

- `record(..., step=...)`
- `get_step_dim_matrix(name, "batch_var")`

### 模式 D：做分支关系分析

适合：

- 看 `innovation` 和 `context` 是否冗余
- 看 `shared` 和 `specific` 是否过于相似

用法：

- `configure_relations`
- `get_relation_stats`

---

## 13. 当前实现的边界

有几件事要明确：

- `TensorObservatory` 不保存全量原始张量
  - 只保存窗口统计和 sketch
- 关系分析不是精确全量分析
  - 是基于 sketch 的近似分析
- 它默认只负责“文本和 history”
  - 不负责自动实验管理
- 它不知道哪些指标对你最重要
  - 仍然需要模型作者自己选择监控对象

这正是它的设计取向：

- 足够轻
- 足够通用
- 足够容易接到任意模型里

---

## 14. 推荐的接入规范

为了后续不同模型的日志能共用同一套阅读方式，建议名字按下面约定来：

- `shared_*`
  - 共享表示、共享 gate、共享 logits
- `specific_*`
  - 专属表示、专属 gate、专属 logits
- `*_weighted`
  - gate 加权后的结果
- `*_feature_importance`
  - 通常是 `abs(feature)` 或 fluctuation 强度
- `*_code_usage`
  - VQ / 路由槽位使用率

这样后面的人拿到新模型日志时，几乎不用重新猜每个名字是什么意思。

---

## 15. 一个完整的实战模板

下面这个模板适合直接复制进模型里。

```python
from betterbole.utils.observatory import (
    RelationOptions,
    TensorDisplayConfig,
    TensorMonitorOptions,
    TensorObservatory,
)


def build_observatory():
    obs = TensorObservatory(
        window_size=50,
        relation_options=RelationOptions(
            enabled=True,
            rank=8,
            max_pairs=12,
            names=(
                "shared_hidden",
                "specific_hidden",
                "innovation_hidden",
                "context_hidden",
            ),
        ),
    )

    feature_options = TensorMonitorOptions(
        display=TensorDisplayConfig(
            show_global_summary=True,
            show_per_dim=True,
            max_display_dims=16,
            topk_display_dims=8,
            rank_by="variance",
        )
    )

    gate_options = TensorMonitorOptions(
        display=TensorDisplayConfig(
            show_global_summary=True,
            show_per_dim=True,
            max_display_dims=4,
            topk_display_dims=4,
            rank_by="variance",
        )
    )

    for name in [
        "shared_hidden",
        "specific_hidden",
        "innovation_hidden",
        "context_hidden",
    ]:
        obs.register(name, feature_options)

    for name in [
        "shared_gate_weights",
        "specific_gate_weights",
    ]:
        obs.register(name, gate_options)

    return obs


def record_debug(obs, debug, step):
    obs.record("shared_hidden", debug["shared_hidden"], step=step)
    obs.record("specific_hidden", debug["specific_hidden"], step=step)
    obs.record("innovation_hidden", debug["innovation_hidden"], step=step)
    obs.record("context_hidden", debug["context_hidden"], step=step)
    obs.record("shared_gate_weights", debug["shared_gate_weights"], step=step)
    obs.record("specific_gate_weights", debug["specific_gate_weights"], step=step)


def print_debug(obs):
    print(
        obs.get_window_stats(
            names=[
                "shared_hidden",
                "specific_hidden",
                "innovation_hidden",
                "context_hidden",
                "shared_gate_weights",
                "specific_gate_weights",
            ],
            include_relations=True,
            relation_names=[
                "shared_hidden",
                "specific_hidden",
                "innovation_hidden",
                "context_hidden",
            ],
        )
    )
```

---

## 16. 相关文件

- 主类
  - [collector.py](D:\pyprojects\recommend-study\studybole\src\betterbole\utils\observatory\collector.py:1)
- 配置
  - [config.py](D:\pyprojects\recommend-study\studybole\src\betterbole\utils\observatory\config.py:1)
- 分析函数
  - [analysis.py](D:\pyprojects\recommend-study\studybole\src\betterbole\utils\observatory\analysis.py:1)
- 谱分析
  - [metrics/spectral.py](D:\pyprojects\recommend-study\studybole\src\betterbole\utils\observatory\metrics\spectral.py:1)
- 文本渲染
  - [formatting.py](D:\pyprojects\recommend-study\studybole\src\betterbole\utils\observatory\formatting.py:1)

---

## 17. 一句话总结

如果只记一条：

`TensorObservatory` 的正确使用方式不是“把所有张量都丢进去”，而是“挑几个最关键的表示，在训练关键节点持续记录，然后用窗口统计、关系分析和 history 去回答具体结构问题”。
