# DiffMSR Backbone 可插拔化方案

## 目标

把当前 `DiffMSRIdModel` 里写死的

- `self.shared_bottom(...)`
- `self.domain_heads[...]`

改造成**可切换 backbone** 的结构，并且做到：

1. **不破坏现有实验逻辑**
   - stage0 / stage1 / stage2 / stage3 / stage4 的训练流程不变
   - 现有 diffusion / classifier / augment 逻辑不变
2. **可以通过参数切换 backbone**
   - 类似 `backbone_name="sharedbottom" | "mmoe" | "ple" | ...`
   - backbone 参数通过 `backbone_params` 传入
3. **与当前 DiffMSR 的输入输出兼容**
   - backbone 统一接口：`forward(self, x, domain_ids)`
   - `x` 是拼好的 `[domain_e, user_e, item_e]`
   - `domain_ids` 是当前 batch 的域 id
4. **先兼容当前实现，再扩展到更多实验**
   - 先支持 DiffMSR 的 pointwise CTR 场景
   - 保持 `embed_user_item_pair / embed_triplet / diffusion` 相关代码不动


---

## 推荐总体设计

### 1）引入抽象基类 `MSRBackbone`

建议新增目录：

- `src/betterbole/models/backbone/msr/base.py`
- `src/betterbole/models/backbone/msr/factory.py`
- `src/betterbole/models/backbone/msr/*.py`

核心抽象：

```python
class MSRBackbone(nn.Module, ABC):
    output_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        \"\"\"Return hidden representation of shape [B, output_dim].\"\"\"
```

这里 **不让 backbone 直接输出 logit**，而是统一输出隐藏表示，再由 `DiffMSRIdModel` 原有的 `domain_heads` 输出 logit。

这样做的原因：

1. **最小改动现有逻辑**
   - 当前 `DiffMSRIdModel` 的 head 路由逻辑已经稳定
   - stage0/stage1/stage4 的 loss 与评估逻辑都依赖这个路由
2. **不影响 diffusion / classifier**
   - stage2 / stage3 依赖的是 `user-item pair embedding`
   - 与 backbone 选择本质解耦
3. **更容易统一 8 个 backbone**
   - 很多 reference 实现自带 tower / sigmoid
   - 迁移到 DiffMSR 时，只保留“backbone representation”部分即可


---

## `DiffMSRIdModel` 未来接口建议

当前关键路径：

```python
features = torch.cat([domain_e, user_e, item_e], dim=-1)
shared_rep = self.shared_bottom(features)
logits = self.domain_heads(...)
```

建议改成：

```python
features = torch.cat([domain_e, user_e, item_e], dim=-1)
shared_rep = self.backbone(features, domain_ids)
logits = self.domain_heads(...)
```

也就是：

- `self.shared_bottom` 改为 `self.backbone`
- `self.backbone` 满足统一签名：
  - `forward(self, x, domain_ids)`

### 推荐保留的 `DiffMSRIdModel` 结构

- `user_emb`
- `item_emb`
- `domain_emb`
- `embed_user_item_pair(...)`
- `embed_triplet(...)`
- `map_domain_ids_to_head_indices(...)`
- `domain_heads`
- `calculate_loss(...)`
- `predict(...)`

### 推荐新增参数

在 `DiffMSRIdModel.__init__` 或配置里新增：

```python
backbone_name: str = "sharedbottom"
backbone_params: dict | None = None
```

并通过工厂函数构造：

```python
self.backbone = build_msr_backbone(
    name=backbone_name,
    input_dim=embedding_size * 3,
    num_domains=len(domain_head_ids),
    default_hidden_dim=...,
    **(backbone_params or {})
)
```


---

## 8 个 backbone 的接入原则

目标模型：

1. `SharedBottom`
2. `MMoE`
3. `PLE`
4. `STAR`
5. `M2M`
6. `EPNet`
7. `PPNet`
8. `M3oE`

### 统一原则

**都改造成“返回 representation，不直接返回概率”的版本。**

也就是：

- reference 项目中的 `tower` / `sigmoid`
- 在 DiffMSR 版本里尽量拆掉或只保留到最后一层隐藏层

因为 DiffMSR 当前已经有：

- domain-specific `head`
- BCE with logits
- per-domain checkpoint selection

所以 backbone 最好只负责：

- shared / domain-aware feature transformation


---

## 各模型建议映射

### A. SharedBottom

**最容易对齐当前实现。**

#### DiffMSR 版定义

- 输入：`x: [B, D]`
- 输出：`h: [B, H]`

#### 实现策略

可以直接视作：

- 共享 MLP bottom
- 不含 tower

当前 `self.shared_bottom` 本质就是它的最简版本。

#### 迁移建议

- 首个默认 backbone 就用这个
- 作为所有新 backbone 的行为对照组


### B. MMoE

你本地已经有：

- `src/betterbole/models/backbone/mmoe.py`

#### 建议

不要重新照搬 reference；优先复用你本地实现思路。

#### 适配方式

当前你本地 `SingleLayerMMoE.forward(x, domain_ids)` 直接输出 domain tower logits。

建议拆成两层：

1. `MMoEBackbone.forward(x, domain_ids) -> pooled_rep`
2. `DiffMSRIdModel.domain_heads` 再输出 logits

也就是把你本地 `towers` 从 backbone 内去掉，或者提供一个 `return_representation=True` 模式。


### C. PLE

你本地已经有：

- `src/betterbole/models/backbone/ple.py`

#### 建议

同 MMoE，优先复用本地版本，不直接引用 reference 代码。

#### 适配方式

把当前：

- `PLE.forward(x, domain_ids) -> tower logit`

改成：

- `PLEBackbone.forward(x, domain_ids) -> final expert state`

最后交给 `DiffMSRIdModel.domain_heads`。


### D. STAR

你本地已经有：

- `src/betterbole/models/backbone/star.py`

#### 建议

优先复用本地实现。

#### 适配方式

当前 `STAR.forward(x, domain_ids)` 也是直接走 tower。

建议拆成：

- `STARBackbone.forward(x, domain_ids) -> domain-conditioned hidden state`

然后保留 DiffMSR 的 head。


### E. M2M

reference 里的 `M2M` 比较重，包含：

- transformer
- scenario/task meta attention
- meta tower

#### 风险

它的 reference 版本默认假设：

- 有额外的 `domain feature embedding`
- 有其自己的 tower 输出结构

#### DiffMSR 适配建议

在 DiffMSR 中把它简化为：

- 输入 `x` 作为统一 dense feature
- `domain_ids` 经 `nn.Embedding(num_domains, domain_emb_dim)` 形成 scenario/task condition
- 保留 M2M 的 meta-mixing 思路
- **去掉最终输出 tower**，返回 `rep`

#### 结论

M2M **建议重写一版轻量 DiffMSR 适配版**，不要直接硬搬 reference。


### F. EPNet

reference 的 `EPNet` 更像：

- scenario 特征 + agnostic 特征
- 通过 gate 调制 agnostic 特征

#### 风险

DiffMSR 当前只有一个拼好的 `x`

```python
x = concat(domain_e, user_e, item_e)
```

并没有天然的：

- `sce_features`
- `agn_features`

#### DiffMSR 适配建议

定义固定划分：

- scenario branch：`domain_e`
- agnostic branch：`concat(user_e, item_e)`

于是：

```python
sce_x = domain_e
agn_x = torch.cat([user_e, item_e], dim=-1)
```

EPNet 在 DiffMSR 中变成：

- 用 `domain_e` 给 `user/item` 分支打门控
- 输出 gated representation

#### 结论

EPNet **建议重写一版 DiffMSR-friendly 版本**。


### G. PPNet

reference 的 `PPNet` 也强依赖：

- id features
- agnostic features
- domain tower

#### DiffMSR 适配建议

与 EPNet 一样，做固定语义映射：

- id branch：`domain_e`
- agnostic branch：`concat(user_e, item_e)`

但在 DiffMSR 版里：

- 去掉最终 domain tower 输出概率的部分
- 只保留 gated progressive representation

#### 结论

PPNet **建议重写一版 DiffMSR 适配版**。


### H. M3oE

reference 的 `M3oE` 本质上是：

- STAR + MMoE + domain/task expert weighting

#### 风险

结构复杂，直接搬 reference 很容易把：

- 输出层
- 温度退火
- 任务维

一起耦进去，导致 DiffMSR 接口变脏。

#### 建议

M3oE 也采用**适配重写**：

- 保留核心模块思想
- 统一输出 `rep`
- 把 task 概念收缩为单 CTR task
- domain 作为唯一 scenario 维度


---

## 我建议的实现方式：**以 BetterBole 为主，选择性重写**

### 不建议的方案

把整个：

- `reference-projects/Scenario-Wise-Rec-main/scenario_wise_rec`

原封不动搬进 `src/`

原因：

1. reference 模型默认输入接口与 DiffMSR 不同
2. 大多数模型自带 embedding / tower / sigmoid
3. 直接 vendoring 会形成两套风格
4. 后续 debug 会比较痛苦

### 更推荐的方案

#### 第 1 类：复用你已有实现

- SharedBottom
- MMoE
- PLE
- STAR

#### 第 2 类：按 DiffMSR 语义重写适配版

- M2M
- EPNet
- PPNet
- M3oE

这样能保证：

- 统一接口
- 统一输出维度
- 统一与 `DiffMSRIdModel` 的拼接方式


---

## 推荐目录结构

建议新增：

```text
src/betterbole/models/backbone/msr/
├── __init__.py
├── base.py
├── factory.py
├── sharedbottom.py
├── mmoe.py
├── ple.py
├── star.py
├── m2m.py
├── epnet.py
├── ppnet.py
└── m3oe.py
```

### `base.py`

- `MSRBackbone`
- `MSRBackboneOutput`（可选）

### `factory.py`

提供：

```python
def build_msr_backbone(
    name: str,
    input_dim: int,
    num_domains: int,
    **kwargs,
) -> MSRBackbone:
    ...
```

### 各子模块

每个模块只暴露一个 backbone 类，例如：

- `SharedBottomBackbone`
- `MMoEBackbone`
- `PLEBackbone`
- `STARBackbone`
- `M2MBackbone`
- `EPNetBackbone`
- `PPNetBackbone`
- `M3oEBackbone`


---

## `DiffMSRIdModel` 预计改动点

### 当前

```python
self.shared_bottom = build_mlp(...)
...
shared_rep = self.shared_bottom(features)
```

### 未来

```python
self.backbone = build_msr_backbone(...)
...
shared_rep = self.backbone(features, domain_ids)
```

### 需要同步改的函数

1. `shared_representation_from_embeddings(...)`
2. `logits_from_embeddings(...)`
3. `configure_stage4_trainability(...)`

其中 `configure_stage4_trainability(...)` 需要从：

- freeze `shared_bottom`

改成：

- freeze `backbone`

或者更细粒度：

- `backbone.freeze_stage4_trainability(...)`


---

## 配置层建议

在 `settings.py` 中增加：

```python
backbone_name: str = "sharedbottom"
backbone_params: dict | None = None
```

例如：

```python
cfg = build_settings(
    custom="msr-mmoe",
    backbone_name="mmoe",
    backbone_params={
        "expert_dims": (256, 256),
        "num_experts": 4,
    },
)
```

或者：

```python
cfg = build_settings(
    custom="msr-ple",
    backbone_name="ple",
    backbone_params={
        "n_level": 2,
        "n_expert_specific": 2,
        "n_expert_shared": 2,
        "expert_dims": (256, 256),
    },
)
```

### 兼容策略

如果 `backbone_name == "sharedbottom"`：

- 默认行为应与当前 `self.shared_bottom` 版本尽量一致

这点非常重要，因为它是验证“没有改坏原始逻辑”的基线。


---

## 推荐分阶段落地

### Phase 1：框架搭好，不碰原始行为

目标：

- 新增 `MSRBackbone`
- 新增 `factory`
- 新增 `SharedBottomBackbone`
- 让默认 `sharedbottom` 跑出来与当前版本尽量一致

验收标准：

- 相同 seed 下，stage0/stage1/stage4 指标波动很小

### Phase 2：迁移你本地已有 backbone

迁移：

- MMoE
- PLE
- STAR

验收标准：

- 能通过配置切换
- 日志、checkpoint、experiment_record 保持完整

### Phase 3：实现 reference 适配版

实现：

- M2M
- EPNet
- PPNet
- M3oE

验收标准：

- 统一接口
- 不引入新的实验脚本特判


---

## 关键兼容性说明

### 1）stage2 / stage3 不应受影响

因为它们依赖的是：

- `embed_user_item_pair`

而不是 `shared_bottom / backbone`

所以 backbone 切换后：

- diffusion 训练逻辑应保持不变
- classifier 训练逻辑应保持不变

### 2）stage1 / stage4 会直接受影响

因为 logits 计算路径依赖：

- backbone representation

所以：

- 需要重点保证 `sharedbottom` 版本对齐旧行为

### 3）stage4 freeze 策略需要重新定义

当前是：

- freeze embeddings
- freeze shared_bottom
- only train heads

未来改成 backbone 后，要明确：

- 是否整个 backbone 都 freeze
- 还是只 freeze 其中 shared trunk
- 某些 backbone 是否保留 domain-specific adaptor 可训练

**建议第一版：完全复用当前策略**

也就是：

- `stage4_freeze_backbone = True`

先保证实验口径一致。


---

## 推荐的最终接口示例

### `DiffMSRIdModel`

```python
model = DiffMSRIdModel(
    manager,
    domain_head_ids=...,
    embedding_size=cfg.emb_dim,
    backbone_name=cfg.backbone_name,
    backbone_params=cfg.backbone_params,
)
```

### backbone factory

```python
backbone = build_msr_backbone(
    name="ple",
    input_dim=emb_dim * 3,
    num_domains=len(domain_head_ids),
    **backbone_params,
)
```

### backbone forward

```python
h = backbone(x, domain_ids)   # [B, H]
logits = domain_heads(h, domain_ids)
```


---

## 我建议的实施顺序

### 第一轮（最稳）

1. 先落 `MSRBackbone` 抽象
2. 实现 `SharedBottomBackbone`
3. 让 `DiffMSRIdModel` 支持 `backbone_name="sharedbottom"`
4. 确认新旧脚本行为基本一致

### 第二轮

5. 迁移 `MMoE / PLE / STAR`

### 第三轮

6. 再做 `M2M / EPNet / PPNet / M3oE`


---

## 我个人推荐的最终决策

### 结论

我建议采用：

- **`MSRBackbone` 抽象类**
- **`forward(self, x, domain_ids)` 统一接口**
- **backbone 只返回 representation**
- **domain head 继续保留在 `DiffMSRIdModel`**
- **先复用 BetterBole 现有的 SharedBottom/MMoE/PLE/STAR**
- **再为 M2M/EPNet/PPNet/M3oE 写 DiffMSR 适配版**

这是我认为**最稳、最少破坏现有逻辑、也最利于后续参数切换**的方案。


---

## 下一步我建议我来做什么

如果你认可这个方案，下一步我建议直接开始做：

1. 新建 `src/betterbole/models/backbone/msr/base.py`
2. 新建 `src/betterbole/models/backbone/msr/factory.py`
3. 先落 `SharedBottomBackbone`
4. 把 `DiffMSRIdModel` 改成支持 `backbone_name/backbone_params`
5. 先保证 `sharedbottom` 跑通

也就是说，**先做一版最小闭环**，不一口气把 8 个都上完。
