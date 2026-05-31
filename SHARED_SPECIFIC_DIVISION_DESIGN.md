# Shared / Specific Division Design Notes

这份文档的目标不是直接给出最终模型，而是把当前已经看清楚的问题、`shavq` 与 `ple_ob` 的启发，以及下一阶段应该逐步验证的训练约束整理清楚。

核心判断只有一句：

**仅靠“共享专家和专属专家的输入不同”还不够，必须通过显式训练约束让它们知道自己应该分工。**

---

## 1. 当前已经看清楚的事情

### 1.1 原始 PLE 的问题不是简单的 capacity 不够

从 `ple_ob` 的 observatory 日志看，原始 PLE 的结构问题主要是：

- `shared` 路低秩
  - `shared_hidden` 的有效秩很低，`top1_energy` 很高
  - 说明 shared 表征被压缩到少数主方向
- `task gate` 接近软平均
  - `task_gate_ent` 长期接近均匀分布熵
  - `task_shared_mass` 接近一半
  - 说明 task gate 没有形成明确路由
- `shared gate` 不是真的在用 shared experts
  - `shared_gate_shared_mass` 很低
  - 说明 shared gate 实际上更偏向吃 specific experts
- `specific` 更像 residual correction
  - `selected_specific_mean_by_layer` 与 `task_shared_delta_by_layer` 的 `linear_cka` 非常高
  - 说明 specific 并不像独立专家，更像在 shared 基底上补差值

所以原始 PLE 的问题不是：

- 参数太少
- 训练不够久

而是：

- **shared / specific 分工没有被训练机制真正建立起来**

### 1.2 SHAVQ 给出的启发是“改输入有用，但不够”

`shavq` 系列和原始 PLE 最大的区别之一，是 shared 与 specific 的输入不再完全相同。

在 `shavq_v4` 的观测里可以看到：

- `shared_hidden` 本身是健康的
  - `eff_rank` 高
  - `top1_energy` 不高
- `specific` 路不再完全复制 shared
  - shared / specific 的 relation 并不高
- 说明“输入改变”确实能帮助分工

但 `shavq_v4` 也暴露出新的问题：

- `specific_context_hidden_weighted` 容易塌成单方向
- `specific_innovation_hidden_weighted` 往往主导最终 `specific`
- `shared` 表征虽然健康，但 logit 侧话语权偏低

这说明：

1. 改输入确实比“所有专家吃同样的输入”更有机会学出分工。
2. 但**只改输入仍然不够**。
3. 如果没有显式训练约束，某些分支仍然会退化为：
   - 共享分支只学一个粗糙基底
   - 专属分支只学一条最容易的修正路径

---

## 2. 为什么“输入不同”不等于“自然分工”

很多时候会有一个直觉：

- 共享专家输入共享特征
- 专属专家输入专属特征
- 那么网络自己应该会学出分工

这个直觉只对了一半。

### 2.1 同样的主体结构会把不同输入也映射到相似角色

如果：

- shared expert 是 MLP
- specific expert 也是 MLP
- 两边都只受最终 CTR loss 驱动

那么优化器会优先找到“最容易降低 loss 的解”，而不是“最符合我们语义期待的解”。

于是很容易出现：

- shared 学到一个低秩的大底座
- specific 学到 shared 的偏移修正
- gate 只做平均混合

这在数学上完全可行，在优化上也很舒服，但语义上没有真正分工。

### 2.2 CTR supervision 本身不关心“谁负责什么”

CTR 只关心：

- 最终 logit 对不对

它并不关心：

- shared 表征是否纯净
- specific 是否真的在学 domain-private 信息
- 专属分支的两路是否彼此互补

所以如果不额外施加结构性约束，模型自然会倾向于：

- 谁容易学，谁多学一点
- 谁不必要，谁就退化

### 2.3 没有分布差异约束，shared / specific 很容易语义混叠

如果我们不要求：

- shared 输出的 CTR 分布与 specific 输出的 CTR 分布显著不同
- shared 表征对 domain 变化不敏感
- specific 表征对 domain 变化敏感且有针对性

那么 shared / specific 最终就可能只是两种参数化方式的同类函数近似器。

---

## 3. 当前的设计方向判断

下一阶段不应该继续把主要精力放在：

- 更细的调参
- 更深的 MLP
- 更小的温度

而应该放在：

- **用训练机制把分工“教”给网络**

这类训练机制大致分三层：

1. shared / specific 的输出分布要被拉开
2. shared / specific 面对 domain 切换时的响应方式要不同
3. 专属分支内部也要继续细分角色，而不是让一条 easiest path 独大

---

## 4. 可以逐步验证的训练约束

下面这些不是要求一次全上，而是建议按顺序逐步实验。

### 4.1 Shared / Specific CTR Distribution Separation Loss

目标：

- 让共享专家输出的 CTR 贡献和专属专家输出的 CTR 贡献在 batch 层面显著不同

直觉：

- 如果两者的 logit 分布太像，说明它们没有在干不同的事情

可以约束的对象：

- `shared_logits`
- `specific_logits`
- 或者它们的分桶 / 排序 / margin 结构

可以尝试的形式：

1. 均值 / 方差差异约束
   - 太弱，只适合做最低阶尝试
2. 分布距离约束
   - 比如 batch 内的 Wasserstein / MMD / JS
3. 排名差异约束
   - 约束 shared 更像“基础排序”
   - specific 更像“局部重排”

设计原则：

- 不是盲目让两者完全不同
- 而是让它们在功能上形成稳定差异

更合理的语义是：

- shared logits: 粗粒度、跨场景一致的基础 CTR 偏好
- specific logits: 场景化修正、局部抬升/压制

### 4.2 Shared Invariance Under Domain Switch

目标：

- shared 表征在切换 domain id 时尽量稳定

实现直觉：

- 冻结 shared 分支或在 stop-grad 条件下，构造同一输入样本的“虚拟 domain 切换”
- shared 输出不该因为 domain id 切换而显著波动

形式上可以做：

- 同一个 `x`
- 替换不同 `domain_id`
- 比较 `shared_hidden` / `shared_logits`

希望看到：

- shared 分支对 domain 切换不敏感
- specific 分支对 domain 切换敏感

这其实是在给 shared / specific 建立一个非常直接的角色定义。

### 4.3 Specific Counterfactual Sensitivity Loss

目标：

- 当切换当前 domain id 后，specific 分支必须给出有意义的差异化响应

这条和上一条配对使用最有意义。

思路：

- 冻结 shared 分支
- 对同一个输入样本，切换 `domain_id`
- 要求不同 specific expert 或不同 domain head 的输出产生足够差异

这个约束的意义是：

- 专属分支不能只是 shared 的小修小补
- 它必须真正携带“当前 domain 的选择性增益”

### 4.4 Shared Adversarial Domain Confusion

目标：

- 共享专家输出难以被判别出来自哪个 domain

这是更经典也更强的一类 shared 纯化约束。

好处：

- 直接打击“共享表示中残留大量 domain-private 污染”

风险：

- 如果做得太强，可能把对 CTR 真的有用的场景差异也抹掉

所以建议：

- 只作为 shared purity 的辅助项
- 不要一开始就给太大权重

### 4.5 Orthogonality / Subspace Decoupling

目标：

- shared 与 specific 在表征子空间上不要高度重叠

可做的对象：

- `shared_hidden` vs `specific_hidden`
- `shared_logits` vs `specific_logits`
- shared 主子空间 vs specific 主子空间

意义：

- 避免“两个分支看起来不同，实际上在学同一方向”

这类 loss 和 observatory 非常配，因为我们已经有：

- `linear_cka`
- `subspace_mean_cos`
- `eff_rank`

可以直接用来验证它是不是真的把子空间拉开了。

---

## 5. 下一步专属专家的角色重构

你提的这条很关键：

> 我设计两种专属专家，一种是负责处理特征中只有在目标场景能看到的信息，另一种则是负责处理共享信息在目标场景上的独特增益。

这是比“专属专家就是 residual”更强的语义设计。

可以把 specific 分支明确拆成两类角色：

### 5.1 Domain-Private Expert

职责：

- 处理只有目标场景能看到的信息
- 或者处理只有在该场景中才有意义的信息

典型来源：

- 场景独占特征
- 场景专属曝光机制
- 场景专属内容供给模式
- 场景专属行为反馈

它的语义更接近：

- “这个场景自己独有的东西”

### 5.2 Shared-Gain Expert

职责：

- 处理共享信息在当前场景的额外增益
- 不是重学 shared
- 而是建模 “shared interest × current domain” 的交互放大/抑制

它的语义更接近：

- “这份共享兴趣在当前场景里值不值得被抬高”

这类 expert 最重要的不是看 raw private feature，而是看：

- shared hidden
- shared logits
- shared 与 domain context 的交互
- shared 对当前 domain 的 counterfactual gain

### 5.3 这两类专属专家不能再吃完全同质的输入

如果这两类 expert 仍然吃几乎相同的输入，再加上相同主体结构，最后还是会退化回“谁先学会 easiest path 谁统治”。

因此应该从输入层面就做区分：

- Domain-Private Expert
  - 更偏向 private-only / scene-only / residual-only 输入
- Shared-Gain Expert
  - 更偏向 shared summary + domain context + gain signals

这件事从 `shavq_v4` 已经得到了一点启发：

- 改输入是有用的
- 但还要配上 loss，才能防止一条路退化

---

## 6. 一个更合理的 shared + specific 信息流想象

如果后面重新设计结构，我更倾向于下面这个语义：

### 6.1 Shared Branch

输入：

- 尽量 domain-invariant 的底层表示
- 或 VQ / prototype / consensus 表征

职责：

- 提供基础 CTR prior
- 学跨场景稳定的排序规则

训练约束：

- domain confusion
- domain switch invariance
- 低污染 shared purity loss

### 6.2 Specific Branch A: Domain-Private

输入：

- 只有当前场景能看到的信息
- 或 shared 无法解释的 residual/private 输入

职责：

- 建模场景独占信息

训练约束：

- 对 domain switch 敏感
- 与 shared 子空间解耦

### 6.3 Specific Branch B: Shared-Gain

输入：

- shared hidden
- shared logits
- domain context
- domain-conditioned interaction features

职责：

- 建模共享兴趣在当前场景的增益/折损

训练约束：

- 与 Domain-Private 分支保持区分
- 只做 gain，不要重学 shared base

### 6.4 融合层

不要只是简单加和，而是要反映语义：

- `final = shared_base + private_correction + gain_correction`

如果需要 gate，它也应该服务于语义选择：

- 当前样本更依赖 private information
- 还是更依赖 shared gain

而不是仅仅学一个黑盒 softmax 混合器。

---

## 7. 为什么这条路线值得继续

因为目前两个系列给出的结论是互补的：

- `ple_ob` 告诉我们
  - 输入相同 + 无约束时，shared / specific 很容易混叠
- `shavq` 告诉我们
  - 输入差异化确实能帮助分工
  - 但没有显式约束时，specific 内部仍会退化成单一路

这正好指向一个中间结论：

> **结构上要先让输入不同，训练上再强制它们的功能不同。**

只有这两件事同时做，shared / specific 的语义分工才更有机会真正成立。

---

## 8. 下一步建议的实施顺序

不要一口气把所有想法全塞进模型里。建议按下面顺序一步步验证：

1. 先做最小版 shared / specific 分布拉开 loss
   - 目标是验证“只加分工约束”是否有明显效果
2. 再做 domain switch invariance / sensitivity 成对实验
   - shared 稳定
   - specific 敏感
3. 再把 specific 正式拆成两类角色
   - Domain-Private
   - Shared-Gain
4. 最后再考虑是否叠加 domain adversarial、子空间解耦等更强约束

这样每一步都能回答一个明确问题，而不是把问题和答案混在一起。

---

## 9. 当前文档对应的结论摘要

如果压缩成最短版本，就是下面这几条：

1. 仅靠输入不同，不足以保证分工。
2. 仅靠最终 CTR loss，网络不会自动学出我们想要的 shared / specific 语义。
3. `shavq` 证明“改输入”是有帮助的。
4. `ple_ob` 证明“没有显式约束时，shared / specific 会混叠”。
5. 下一阶段必须引入 batch 级或 counterfactual 级 loss，显式拉开 shared 与 specific 的功能分布。
6. 专属专家内部也应继续拆分成：
   - 处理场景独占信息的 expert
   - 处理共享信息场景增益的 expert

这份文档的任务到这里为止：先把方向说清楚，后面的实现再一步一步做。

---

## 10. I2MoE 给我们的直接启发

这次翻 `reference-projects/I2MoE-main`，最值得记住的不是某个具体网络层，而是它**约束专家能力的方式**。

### 10.1 I2MoE 不是“先有专家，再看它们学出什么”

它是反过来的：

- 先预定义专家角色
- 再为每种角色设计专门的弱监督约束

在 `InteractionMoE.py` 里，专家不是完全对称的一组 anonymous experts，而是固定分成：

- `num_modalities` 个 `uniqueness` experts
- 1 个 `synergy` expert
- 1 个 `redundancy` expert

也就是说，它从一开始就不接受“所有专家都做同一件事，只靠 gate 去分流”的设定。

这点和我们现在看到的 PLE / SHAVQ 现象很一致：

- 如果不给专家角色定义
- 它们最后大概率就会收敛到几个 easiest path

### 10.2 它约束专家能力的关键手段是 perturbation-based weak supervision

I2MoE 的每个专家都会拿到：

- 一次完整输入的输出 `anchor`
- 多次“某个模态被替换掉”之后的输出

这里的替换在主实现里是：

- 把某个模态直接替换成随机向量

然后不同类型的专家，用不同的 loss 去约束这些输出之间的关系。

#### Uniqueness Expert

目标：

- 当某个特定模态被破坏时，输出应该显著变化
- 当其他模态被破坏时，输出不应该变化得那么剧烈

代码里的实现方式是 triplet loss：

- `anchor = 完整输入输出`
- `neg = 目标模态被替换后的输出`
- `pos = 其他模态被替换后的输出`

于是它逼这个 expert 学成：

- “我主要依赖某一个特定来源的信息”

#### Synergy Expert

目标：

- 只要破坏任一模态，输出都应该受影响

实现方式：

- 计算完整输出与各个扰动输出的余弦相似度
- 把这些扰动输出当成 `negatives`

直觉上，它逼 expert 学成：

- “我依赖的是多源协同，缺一块就不对了”

#### Redundancy Expert

目标：

- 即使某个模态被破坏，输出也应尽量稳定

实现方式：

- 完整输出与各个扰动输出做余弦约束
- 惩罚它们之间的差异

这会逼 expert 学成：

- “我负责建模多源之间可互相替代、比较稳健的共识部分”

### 10.3 Reweighting 头负责“按样本选谁更有用”，但不负责定义专家语义

I2MoE 里还有一个 `MLPReWeighting`：

- 输入是样本的模态表示
- 输出是各个 interaction experts 的权重

然后最终输出是：

- 各专家 logits 的加权和

这部分很重要，但它的职责是：

- **在样本层面选择专家贡献比例**

而不是：

- **负责把专家训练出不同语义**

专家语义本身，是前面的 interaction losses 决定的。

这也是我们现在对 gate 的判断：

- gate 可以放大已经存在的分工
- 但 gate 不是分工的来源

### 10.4 它的总 loss 结构也很值得借鉴

在 `imoe_train.py` 里，训练目标基本是：

- `task_loss`
- `+ interaction_loss_weight * interaction_loss`
- `+ gate_loss_weight * gate_loss`（如果底层 fusion 本身是 sparse gate）

这说明它把目标拆成三层：

1. 任务正确性
2. 专家角色正确性
3. 稀疏门控本身的训练稳定性

相比之下，我们现在的大问题是：

- 有第 1 层
- 偶尔有第 3 层
- 但第 2 层几乎没有

所以 shared / specific 常常学成“都能做一点 CTR”的模糊分工。

### 10.5 这套思想怎么映射到我们的 shared / specific 设定

I2MoE 的核心启发不是“也做 uniqueness/synergy/redundancy”，而是：

- **先定义专家应该对什么扰动敏感**
- **再把这种敏感性写成 loss**

映射到我们的推荐场景，更自然的扰动不是“模态替换”，而是“domain / feature / shared signal 的反事实扰动”。

可以直接对应出下面几类角色。

#### Shared Expert

希望它对什么稳定：

- 切换 `domain_id`
- 屏蔽一部分场景私有信号

不希望它学到什么：

- 过强的 domain-private 偏置

因此可以给它：

- domain switch invariance loss
- domain adversarial confusion
- private feature masking consistency

#### Domain-Private Expert

希望它对什么敏感：

- 当前 domain 的切换
- 当前场景独占特征的屏蔽

不希望它学到什么：

- 跨场景稳定共识

因此可以给它：

- counterfactual domain sensitivity loss
- private masking sensitivity loss

#### Shared-Gain Expert

希望它对什么敏感：

- shared 表示在不同 domain 下的收益变化

不希望它学到什么：

- 完整重学 shared base
- 退化成普通 private residual

因此可以给它：

- `shared_logits(x, d1)` 与 `shared_logits(x, d2)` 的 gain-delta contrast
- 与 Domain-Private 分支的子空间解耦或输出去相关

### 10.6 对我们最关键的结论

I2MoE 证明了一件很重要的事：

> **想让专家能力不同，不能只靠输入不同，也不能只靠 gate 分流，必须给专家设计“它该对什么扰动敏感/不敏感”的监督。**

这和我们已经观察到的现象是完全一致的：

- `ple_ob` 里，大家输入差不多，最后就混叠
- `shavq` 里，输入开始不同了，分工改善，但仍不稳定

因此下一步真正该补上的，不是更多调参，而是：

- shared / specific 的反事实扰动训练
- shared / specific 的角色约束 loss
- specific 内部分支的能力边界约束
