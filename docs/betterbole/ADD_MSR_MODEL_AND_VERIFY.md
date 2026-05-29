# 新增一个 MSR Model 并验证实验

> 适用于 betterbole 的多场景推荐模型体系。目标是把一个新的模型包接入 `models/msr/`，然后直接复用现成实验脚本做验证。

## 适用场景

如果你要新增一个 MSR 模型，推荐按下面这个最小闭环走：

1. 在 `src/betterbole/models/msr/` 下新建一个包。
2. 在 `src/betterbole/models/msr/__init__.py` 里注册新模型。
3. 直接复用 `@examples/kuairand-1k/kuairan1k.py` 跑实验。
4. 只修改 `cfg.model`，把模型名切到你刚注册的名字。

这个流程不要求你改其余实验参数。`@examples/kuairand-1k/kuairan1k.py` 已经搭好了数据、特征、训练器、评估器和保存路径，通常只需要替换 model 名称即可。

---

## 1. 新建模型包

在 `src/betterbole/models/msr/` 下创建一个新目录，例如：

```text
src/betterbole/models/msr/my_model/
  __init__.py
  model.py
```

建议保持和现有模型一致的结构：

- `model.py` 放模型主体类。
- `__init__.py` 只负责导出公共类名。

如果你的模型还需要额外组件，也可以继续拆分子模块，但对外入口最好统一从 `__init__.py` 暴露。

一个最小模型类通常继承 `MSRModel`，并实现下面这些约定：

- `self.DOMAIN = self.manager.domain_field`
- `self.LABEL = self.manager.label_field`
- `self.input_view = self.omni_embedding.whole`
- `predict(interaction)`
- `calculate_loss(interaction)`

参考现有实现可以直接看：

- [`src/betterbole/models/msr/ple/model.py`](../../src/betterbole/models/msr/ple/model.py)
- [`src/betterbole/models/msr/sharedbottom.py`](../../src/betterbole/models/msr/sharedbottom.py)

---

## 2. 在 `__init__.py` 注册

创建好模型类后，要在 `src/betterbole/models/msr/__init__.py` 里做两件事：

1. `import` 进来。
2. 加入 `MODEL_REGISTRY`。

示例：

```python
from betterbole.models.msr.my_model import MyModel

MODEL_REGISTRY["my_model"] = MyModel
```

更推荐直接按照现有风格一次性维护注册表，例如：

```python
MODEL_REGISTRY = {
    ...
    "my_model": MyModel,
}
```

注册名就是后面实验脚本里要填的 `cfg.model` 值。建议使用小写、下划线风格，避免大小写混淆。

---

## 3. 直接复用 KuaiRand 实验脚本

实验入口已经在这里：

- [`@examples/kuairand-1k/kuairan1k.py`](../../../@examples/kuairand-1k/kuairan1k.py)

这个脚本已经完成了：

- 数据读取和 join
- 特征定义与 `SchemaManager`
- train/valid/test 切分
- `ParquetStreamDataset`
- `EvaluatorManager`
- `Trainer`

因此，验证新模型时一般不需要改实验配置，只要把这里的模型名换掉：

```python
model: str = "ple_version1"
```

改成你注册的新名字，例如：

```python
model: str = "my_model"
```

然后脚本里的这行会自动构建你的模型：

```python
model = build_model(manager, num_domains, cfg.model, embed_dim=16, aux_loss_weight=cfg.aux_loss_weight)
```

只要你的模型已经注册到 `MODEL_REGISTRY`，这里就能直接实例化。

---

## 4. 验证结果

跑实验后，重点看三类输出：

- 训练过程日志，确认模型正常 forward / backward。
- `EvaluatorManager` 的指标输出，确认 AUC / LogLoss 等结果可读。
- 训练是否稳定收敛，确认没有 shape mismatch、domain 索引错误、loss 为 `nan` 等问题。

建议至少做下面这几个检查：

1. 启动后能成功完成 `SchemaManager.fit()` 和 `save_as_dataset()`。
2. `build_model(...)` 能成功返回你的模型实例。
3. 第一轮训练能正常跑完，没有维度错误。
4. 验证阶段能输出整体指标和分场景指标。

如果你是第一次接入新模型，优先保证“能跑通 + 指标能出”，再做结构调参和性能优化。

---

## 5. 最小检查清单

- 新模型包已创建在 `src/betterbole/models/msr/<your_model>/`
- 新模型类已在 `src/betterbole/models/msr/__init__.py` 注册
- `cfg.model` 已改成注册名
- `@examples/kuairand-1k/kuairan1k.py` 可直接运行
- 训练日志和验证指标正常输出

---

## 6. 推荐参考顺序

如果你想快速抄一个可运行模板，建议按这个顺序看：

1. [`src/betterbole/models/msr/base.py`](../../src/betterbole/models/msr/base.py)
2. [`src/betterbole/models/msr/ple/model.py`](../../src/betterbole/models/msr/ple/model.py)
3. [`src/betterbole/models/msr/__init__.py`](../../src/betterbole/models/msr/__init__.py)
4. [`@examples/kuairand-1k/kuairan1k.py`](../../../@examples/kuairand-1k/kuairan1k.py)

这四个文件基本覆盖了“模型定义 -> 注册 -> 实验验证”的完整闭环。
