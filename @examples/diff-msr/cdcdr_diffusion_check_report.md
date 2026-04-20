# CDCDR Diffusion 结构复现检查报告

## 检查对象

- 当前测试文件：`src/betterbole/models/generative/diffusion/test.py`
- 当前相关实现：
  - `src/betterbole/models/generative/diffusion/diffusions.py`
  - `src/betterbole/models/generative/diffusion/base.py`
  - `src/betterbole/models/generative/diffusion/schedulers.py`
- 参考原始实现：`reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py`

## 结论

`test.py` 目前**只复现了原始 CDCDR 扩散部分的一部分骨架**，尤其是：

- `none_embedding`
- `step_mlp`
- `mlp2` 版 `diffu_mlp`
- 训练时 `x_0 + q_sample + 条件丢弃 + 预测 x_0`

但它**没有严格复现原始推理路径**，并且有几个关键的不一致点会直接影响“是否真正在模拟原版 CDCDR”的判断。

如果目标是“结构上尽量贴近原始 `cdcdr.py`”，那么当前版本更接近：

- “把原始扩散块抽成一个通用 diffusion framework 后的近似重写”

而不是：

- “逐行/逐路径保持原始 CDCDR diffusion 行为”

---

## 一、对齐得比较好的部分

### 1. 时间编码 MLP 基本对齐

参考实现中的时间编码：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:273`

当前实现：

- `src/betterbole/models/generative/diffusion/diffusions.py:118`

两边结构都是：

- `SinusoidalPosEmb(256)`
- `Linear(256, embedding_size)`
- `GELU`
- `Linear(embedding_size, embedding_size)`

这一块是对齐的。

### 2. `mlp2` 扩散头结构对齐

参考实现：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:283`

当前实现：

- `src/betterbole/models/generative/diffusion/diffusions.py:125`

都是：

- 输入 `concat(x, h, t)`
- `Linear(3D, 2D) -> GELU -> Linear(2D, D)`

所以如果你要复现的是原版 `diffuser_type == 'mlp2'`，这部分是基本一致的。

### 3. 训练时的“无条件替换”思路是对齐的

参考实现：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:493`

当前实现：

- `src/betterbole/models/generative/diffusion/diffusions.py:148`

两边本质都是：

- 训练时按概率 `p / uncon_p`
- 把条件向量 `h` 替换成 `none_embedding`

只是写法不同：

- 原版用 `add_uncon(h)`
- 现在写进了 `_raw_predict(...)`

语义上是接近的。

---

## 二、最关键的不一致点

### 1. `test.py` 标成 “DDPM 推理”，但实际跑的是 `DDIMScheduler`

`test.py` 中：

- `src/betterbole/models/generative/diffusion/test.py:27`
- `src/betterbole/models/generative/diffusion/test.py:39`

测试文件只实例化了 `DDIMScheduler`，然后把：

- `num_inference_steps=200`

称为“DDPM 推理阶段”。

但这并不等于原始 CDCDR 的 `p_sample(...)`：

- 原始 DDPM 路径：`reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:343`
- 当前实际调用的是：`src/betterbole/models/generative/diffusion/schedulers.py:164`

即使 `DDIMScheduler` 走满 200 步，它仍然是 **DDIM 的 step 公式**，不是原始 `p_sample` 的后验采样公式。

**这个是当前测试文件里最大、最明确的语义错位。**

### 2. 原始推理路径是 `sample_from_noise -> i_sample`，当前测试走的是通用 `denoise`

原始推理入口：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:375`
- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:605`

当前测试入口：

- `src/betterbole/models/generative/diffusion/test.py:49`
- `src/betterbole/models/generative/diffusion/test.py:64`
- `src/betterbole/models/generative/diffusion/base.py:47`

原版是：

- `sample_from_noise(self.denoise_step, self.denoise_uncon, h)`

也就是：

- 明确分成 condition / uncondition 两条 forward 路径
- 明确使用原版手写的 `i_sample(...)`

现在是：

- 用通用 `denoise(...)`
- 在通用框架里做 batch doubling + CFG 融合
- 再交给 scheduler.step

这是一种**框架化重写**，不是原样结构复刻。

### 3. CFG 系数定义不一致：原版 `w` 与当前 `guidance_scale` 不是同一个量

原版 CFG：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:345`
- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:364`

公式是：

- `x_start = (1 + w) * cond - w * uncond`

当前通用 CFG：

- `src/betterbole/models/generative/diffusion/base.py:80`

公式是：

- `output = uncond + s * (cond - uncond)`
- 展开后是 `s * cond + (1 - s) * uncond`

两者等价需要满足：

- `s = 1 + w`

而 `test.py` 里：

- `src/betterbole/models/generative/diffusion/test.py:53`
- `src/betterbole/models/generative/diffusion/test.py:68`

传的是：

- `guidance_scale=2.0`

但原始代码注释里的默认含义是：

- `w = 2`
- 参考：`reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:260`

如果要和原版 `w=2` 对齐，当前通用 CFG 里更接近的应该是：

- `guidance_scale = 3.0`

也就是说，**当前测试的 CFG 强度与原版默认语义不一致**。

### 4. 原版 DDIM 不是“20 步跳采样”，而是 `linespace=100` 的固定稀疏策略

原版：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:259`
- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:315`
- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:380`

关键点：

- `timesteps = 200`
- `linespace = 100`
- `indices = [0, 100, 200]`
- 反推时实际上只有 `3` 个稀疏 step

而 `test.py` 中的“DDIM”是：

- `src/betterbole/models/generative/diffusion/test.py:63`

即：

- `num_inference_steps=20`

这和原版不是一回事。  
原版不是“随便少走一些步”，而是带有**固定步长设计**的特殊采样路径。

### 5. 原版有一个很重要的小 trick：时间嵌入用的是 `0/100/200`，但 DDIM 系数索引实际落在 `0/99/199`

原版：

- 稀疏 alpha 索引构造：`reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:315`
- `indices_now = [0, 99, 199]`：`reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:317`
- 但推理传给 `step_mlp` 的 step 是 `n * linespace`，即 `200/100/0`：`reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:381`

这说明原版存在一个**“时间 embedding 的 step 值”和“DDIM 系数查表索引”并不完全相同**的小 trick / 小不规整。

当前通用 `DDIMScheduler.set_timesteps(...)`：

- `src/betterbole/models/generative/diffusion/schedulers.py:143`

会直接生成统一时间序列，时间 embedding 和 alpha 索引天然绑定成同一个 `timestep`。  
所以这块**没有复现原版那个小细节**。

---

## 三、一个非常容易忽略、但影响很大的问题

### `calculate_loss()` 会把模型置为 train 模式，而 `test.py` 在推理前没有切回 `eval()`

当前通用训练入口：

- `src/betterbole/models/generative/diffusion/base.py:23`
- `src/betterbole/models/generative/diffusion/base.py:36`

`calculate_loss(...)` 内部显式调用了：

- `self.train()`

而 `test.py` 在训练 loss 之后直接调用：

- `src/betterbole/models/generative/diffusion/test.py:49`
- `src/betterbole/models/generative/diffusion/test.py:64`

中间**没有 `model.eval()`**。

这会导致 `_raw_predict(...)` 里的训练期条件丢弃逻辑仍然生效：

- `src/betterbole/models/generative/diffusion/diffusions.py:156`

也就是推理阶段仍可能随机把条件 `y` 替换成 `none_embedding`。

这和原始 `full_sort_predict(...)` 的使用语境完全不同：

- 原始推理是在评估阶段调用，不应该带训练期随机无条件替换。

**这是一个非常关键的不一致点。**  
从结构检查角度，它甚至比“DDPM / DDIM 标签写错”更影响实际行为。

---

## 四、当前测试没有覆盖到的原始结构

### 1. 没覆盖 `diffuser_type`

原版：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:279`

支持：

- `mlp1`
- `mlp2`

当前 `CDCDRMlpDiffusion` 固定就是 `mlp2`：

- `src/betterbole/models/generative/diffusion/diffusions.py:125`

### 2. 没覆盖 `beta_sche`

原版支持：

- `linear`
- `exp`
- `cosine`
- `sqrt`

参考：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:291`

当前 scheduler 只支持：

- `linear`
- `cosine`

当前实现：

- `src/betterbole/models/generative/diffusion/schedulers.py:22`

所以如果你说“复现原始结构”，那现在只覆盖了其中一部分配置空间。

### 3. 没覆盖原始大模型里的联合损失与双域逻辑

原版 diffusion 并不是独立模块单测，它嵌在整个 CDCDR 中：

- `get_user_representation(...)`
- `domain_condition_generator(...)`
- recommendation loss
- diffusion loss
- source / target 双域加权

参考：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:532`

当前 `test.py` 只验证了：

- 随机张量 `pos_item_e`
- 随机张量 `UI_aggregation_e`
- 单独 diffusion loss 和 denoise shape

这足够做**模块 smoke test**，但不够证明“已复现原始结构行为”。

### 4. 没复现原始初始化方式

原版在扩散相关模块建完后会做：

- `self.apply(xavier_normal_initialization)`

参考：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:325`

当前独立扩散模型没有这一步。  
这不一定是 bug，但会让数值分布、训练初期状态和原版不完全一致。

---

## 五、可以视为“等价改写”的地方

以下几处虽然写法不一样，但我认为可以视为合理等价：

### 1. `q_sample` 被抽象成 scheduler 的 `add_noise`

参考：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:331`

当前：

- `src/betterbole/models/generative/diffusion/base.py:40`
- `src/betterbole/models/generative/diffusion/schedulers.py:149`

这是正常的工程化抽象，不算问题。

### 2. `denoise_uncon` 被 `null_y + batch doubling` 替代

参考：

- `reference-projects/cd-cdr/CDCDR/model/cross_domain_recommender/cdcdr.py:391`

当前：

- `src/betterbole/models/generative/diffusion/base.py:73`
- `src/betterbole/models/generative/diffusion/diffusions.py:170`

如果模型处在 `eval()`，且 `null_y` 就是 `none_embedding`，那么这个思路基本是等价的。

---

## 六、额外发现的旁路问题（不是本次主诉，但值得记一下）

`BaseDiffusionModel.reconstruct(...)` 和 `refine(...)` 里调用 `denoise(...)` 时传了不存在的参数：

- `src/betterbole/models/generative/diffusion/base.py:96`
- `src/betterbole/models/generative/diffusion/base.py:129`

传的是：

- `t_start_ratio=...`

但 `denoise(...)` 的签名里没有这个参数：

- `src/betterbole/models/generative/diffusion/base.py:47`

这和本次 `test.py` 检查不是同一个问题，但如果后面继续复用这个通用扩散框架，这里迟早会报错。

---

## 七、我给你的最终判断

如果你的问题是：

> `src/betterbole/models/generative/diffusion/test.py` 是否已经“严格复现”了 `cdcdr.py` 的原始扩散结构？

我的结论是：

- **没有严格复现**

如果你的问题是：

> 它是不是已经抓住了原始 diffusion block 的核心骨架？

我的结论是：

- **抓住了训练骨架和 `mlp2` 网络骨架**
- **但推理路径、CFG 系数语义、稀疏 DDIM 步进策略、eval/train 模式处理，都还没对齐**

## 最值得优先关注的 4 个点

按重要性排序，我认为最关键的是：

1. `test.py` 的“DDPM 推理”其实不是 DDPM，而是 DDIM full-step。
2. 训练 loss 后没有 `model.eval()`，导致推理仍带训练期随机无条件替换。
3. `guidance_scale=2.0` 不等价于原版默认 `w=2`，正确对应关系是 `scale = 1 + w`。
4. 原版 DDIM 是 `linespace=100` 的固定稀疏策略，不是简单 `20 steps`。

## 本次操作说明

- 未修改任何现有代码文件
- 仅新增本报告文件，便于你后续逐项核对
