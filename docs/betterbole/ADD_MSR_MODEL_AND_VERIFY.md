# 新增一个 MSR 模型并验证

这份指南对应当前源码里的多场景模型体系，入口以 [`src/betterbole/models/msr/__init__.py`](../../src/betterbole/models/msr/__init__.py) 为准。

目标是完成一个最小闭环：

1. 新建模型类。
2. 注册到 `MODEL_REGISTRY`。
3. 用当前的 KuaiRand 示例脚本验证能跑通。

## 1. 先理解现有模型的最小形状

绝大多数 MSR 模型都遵循同一套路：

```python
class SomeModel(MSRModel):
    def __init__(self, manager, num_domains, ...):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim
        ...

    def predict(self, interaction):
        ...

    def calculate_loss(self, interaction):
        ...
```

最值得直接参考的现有实现：

- [`src/betterbole/models/msr/sharedbottom.py`](../../src/betterbole/models/msr/sharedbottom.py)
- [`src/betterbole/models/msr/mmoe.py`](../../src/betterbole/models/msr/mmoe.py)
- [`src/betterbole/models/msr/ple/model.py`](../../src/betterbole/models/msr/ple/model.py)

## 2. 新建模型目录

推荐结构：

```text
src/betterbole/models/msr/my_model/
  __init__.py
  model.py
```

`__init__.py`

```python
from .model import MyModel

__all__ = ["MyModel"]
```

`model.py`

```python
import torch
import torch.nn.functional as F

from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components.heads import DomainTowerHead
from betterbole.models.utils.general import MLP


class MyModel(MSRModel):
    def __init__(self, manager, num_domains, hidden_dims=(128, 64), dropout_rate=0.2):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        self.encoder = MLP(self.input_dim, *hidden_dims, dropout_rate=dropout_rate)
        self.head = DomainTowerHead(num_domains, input_dim=hidden_dims[-1])

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        x = torch.flatten(x, start_dim=1)
        domain_ids = interaction[self.DOMAIN].long()
        return x, domain_ids

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        h = self.encoder(x)
        return self.head(h, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return F.binary_cross_entropy_with_logits(logits, labels)
```

## 3. 注册到 `MODEL_REGISTRY`

编辑 [`src/betterbole/models/msr/__init__.py`](../../src/betterbole/models/msr/__init__.py)：

```python
from betterbole.models.msr.my_model import MyModel

MODEL_REGISTRY = {
    ...
    "my_model": MyModel,
}
```

然后你就可以这样构造：

```python
from betterbole.models.msr import build_model

model = build_model(manager, num_domains=5, model_cls="my_model")
```

## 4. `from_manager()` 的约束

`build_model()` 最终会调用 `MSRModel.from_manager()`。它会做两件事：

- 如果你的 `__init__` 接受 `**kwargs`，那所有额外参数都会透传进去。
- 如果你的 `__init__` 只声明了部分参数，未声明的参数会被忽略，并触发 warning。

所以模型构造参数最好写得明确，不要依赖隐式吞参。

## 5. 用当前 KuaiRand 示例做验证

当前更接近源码主路径的实验脚本是：

- [`@examples/kuairand-1k/kuairan1k.py`](../../@examples/kuairand-1k/kuairan1k.py)

验证方式最简单的一种是：

1. 把 `cfg.model` 改成你的注册名。
2. 保持其他训练流程不动。
3. 直接运行脚本。

如果你只想快速切模型，脚本里真正关键的是这一行：

```python
model = build_model(manager, num_domains, cfg.model, aux_loss_weight=cfg.aux_loss_weight)
```

## 6. 第一轮验证至少检查什么

### 结构正确性

- 模型能被 `build_model()` 找到。
- `predict()` 能返回形状为 `[B]` 或 `[B, 1]` 后再 squeeze 成 `[B]` 的张量。
- `calculate_loss()` 能吃下当前 batch 里的 label。

### 数据兼容性

- `interaction[self.DOMAIN]` 对应的 domain 字段存在。
- `self.input_view.embedding_dim` 大于 0。
- 如果模型需要序列特征，不要默认 `omni_embedding.whole` 会自动把序列拼进去；序列通常要通过 `omni_embedding.seq_groups[...]` 显式取。

### 评估兼容性

- 当前 `BaseTrainer.evaluate_epoch()` 会把 `predict()` 的输出直接送到 evaluator。
- 如果你要算 `logloss`，`predict()` 最好返回概率值，或者你自己覆写 `predict_step()`。
- 只算 `auc` 时，raw logits 通常也能用。

## 7. 推荐的调试顺序

1. 先用很小的 batch 跑一个 forward。
2. 再确认 loss.backward() 没有 shape / dtype 错误。
3. 再跑完整个 `train_epoch()`。
4. 最后验证 `evaluate_epoch()` 和 `EarlyStopper` 没问题。

## 8. 一份最小检查清单

- 新模型目录存在且可导入。
- `MODEL_REGISTRY` 已注册。
- `cfg.model` 已切到新名字。
- `build_model(...)` 返回的是你的类实例。
- 第一轮训练和验证都能结束。
- 输出指标字典不为空。
