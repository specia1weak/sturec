# PLE-OB

`ple_ob` 是原始 `ple` 的一个观测版，不改主干训练逻辑，只把中间表示、gate 和专家输出暴露给 `observatory` 做结构分析。

它的目标不是先追 AUC，而是回答这些问题：

- shared expert 是否真的在承载有效表征。
- task gate 是否真的在路由，还是只是接近均匀混合。
- shared gate 是否更偏向少数 expert。
- specific / shared / delta 之间是否存在明显冗余。
- 哪些分支在逐步低秩化。

## 记录了什么

训练中会记录这些关键张量：

- `ple_ob_selected_hidden`
- `ple_ob_shared_hidden`
- `ple_ob_selected_task_outputs_by_layer`
- `ple_ob_shared_outputs_by_layer`
- `ple_ob_selected_specific_mean_by_layer`
- `ple_ob_shared_expert_mean_by_layer`
- `ple_ob_task_shared_delta_by_layer`
- `ple_ob_selected_task_gates_by_layer`
- `ple_ob_shared_gate_weights_by_layer`
- `ple_ob_gate_mass_by_layer`
- `ple_ob_shared_gate_mass_by_layer`

这些张量分别对应：

- 最终被 head 使用的 task 表征
- 每层的 shared 输出
- 每层所有 task 输出
- 每层 selected task 的 specific expert 均值
- 每层 shared expert 的均值
- selected task 与 shared 的差分
- selected task gate
- shared gate
- gate 聚合后的 specific/shared mass

## 当前最重要的观测指标

在 `custom_train_step` 中会打印：

- `feat_ratio`
  - shared hidden 方差占比
- `logit_var`
  - 最终预测 logits 的方差
- `task_gate_ent`
  - selected task gate 的熵
- `shared_gate_ent`
  - shared gate 的熵
- `task_shared_mass`
  - selected task gate 分给 shared experts 的总质量
- `shared_gate_shared_mass`
  - shared gate 分给 shared expert 分量的总质量
- `align`
  - selected hidden 和 shared hidden 的余弦相似度
- `delta_var`
  - task shared delta 的方差
- `low_var(selected/shared)`
  - selected / shared 最终表征中的死维比例

## 目前已经看到的典型现象

基于 2026-05-31 的 KuaiRand 首轮窗口，已经能看到一些比较稳定的结构信号：

### 1. shared hidden 不是塌的，但也不是特别强

在前几个窗口中，`ple_ob_shared_hidden` 的有效秩大约在 1.8 到 2.0 左右，`top1_energy` 大约 0.83 到 0.84。

这说明 shared 路不是死掉了，但表达非常集中，明显偏低秩。

### 2. selected hidden 比 shared hidden 更“像任务表征”

`ple_ob_selected_hidden` 的有效秩大约在 2.9 左右，方差也更高，说明 task side 保留了更多变化。

但它和 shared hidden 的关系也不弱，`linear_cka` 处在 0.4 到 0.6 区间，说明二者并不是完全独立。

### 3. task gate 基本接近均匀

`task_gate_ent` 长期在 0.689 到 0.691 附近，说明 selected task gate 还没有学出特别尖锐的路由。

同时：

- `task_shared_mass` 大约 0.47 到 0.49
- 也就是 task gate 平均有接近一半质量给 shared experts

这通常意味着：

- gate 并没有强烈区分 specific / shared
- 更像是在做软平均

### 4. shared gate 更偏向 specific experts

`shared_gate_shared_mass` 只有大约 0.167 到 0.170。

这说明 shared gate 绝大部分质量并没有给 shared experts，而是落在 specific experts 上。

这点很关键，说明原始 PLE 的 shared 路并不“纯”：

- task gate 在平均用 shared
- shared gate 却主要在吃 specific

这正是 PLE 里很容易出现的结构性混合问题。

### 5. selected specific 和 shared delta 极其相似

`relation_stats` 中，`ple_ob_selected_specific_mean_by_layer` 和 `ple_ob_task_shared_delta_by_layer` 的 `linear_cka` 很高，接近 0.98。

这说明：

- selected task 里真正起作用的变化，基本就是 task 相对 shared 的偏移
- 但这个偏移本身并没有和 shared 形成清晰分离

换句话说，当前 PLE 更像是：

- shared 提供低秩基底
- task 只是在基底上做一个近似线性的修正

而不是一个真正双路互补的结构。

## 这意味着什么

当前 `ple_ob` 暴露出的潜在问题，至少有三类：

1. `shared` 路存在低秩化倾向，容量没完全展开。
2. `task gate` 没有明显形成硬路由，更多是软平均。
3. `shared gate` 对 shared experts 的利用率偏低，结构上并不“共享”。

如果后面继续改 PLE，比较值得尝试的方向是：

- 给 shared 路更清晰的目标，而不是只靠 task loss 间接驱动。
- 增强 gate 的判别能力，避免长期停在接近均匀分布。
- 单独检查 shared experts 的冗余度，确认是否存在多个 expert 学到近似同一方向。

## 怎么跑

训练时直接指定模型名：

```bash
uv run python @examples/kuairand-1k/kuairan1k.py --model ple_ob --log_name ple_ob_eval.log
```

运行后主要看两个地方：

- stdout 里的 `[PLE-OB Recorder]`
- `ctx.recorder.get_window_stats(...)` 打出来的表

## 结论

`ple_ob` 的价值不在于“比原始 PLE 更强”，而在于它能把原始 PLE 的结构问题直接暴露出来。

从当前这轮窗口看，最明显的问题不是 dead code，而是：

- shared 路低秩
- gate 路由不够尖锐
- shared / specific 之间互相渗透太强，分工不清

## Gate Sharpening 试验

为了单独验证“softmax 锐化”这个方向，新增了一个变体：

- `ple_ob_sharp`

它只改 gate 分布公式，不改 PLE 主干。当前实现支持：

- 温度缩放
  - `softmax(logits / temperature)`
- 幂次锐化
  - `softmax(...) ** power` 后再归一化

当前 `ple_ob_sharp` 的默认设定是：

- `task_gate_temperature = 0.7`
- `shared_gate_temperature = 0.7`
- `power = 1.0`

### 当前结论

在 2026-05-31 的一轮 KuaiRand 观测中，轻度温度锐化只带来了非常有限的变化：

- `task_gate_ent`
  - 原始 `ple_ob` 大约 `0.689 ~ 0.691`
  - `ple_ob_sharp` 大约 `0.686 ~ 0.689`
- `shared_gate_ent`
  - 也只是轻微下降
- `task_shared_mass`
  - 仍在 `0.47 ~ 0.49` 一带
- `shared_gate_shared_mass`
  - 仍在 `0.167 ~ 0.170` 附近

这说明：

1. 轻度 temperature sharpening 确实让 gate 分布更尖一点。
2. 但它没有从根本上解决原始 PLE 的软平均问题。
3. shared gate 仍然主要在吃 specific experts，结构问题没有被翻转。

### 解释

这通常意味着：

- 当前 gate logits 本身的可分性就不够强
- 只靠后处理锐化，最多是把已有偏好稍微放大
- 但无法凭空创造“该分给谁”的判别信息

换句话说：

- `sharp softmax` 更像是放大器
- 不是信息来源

### 这个方向下一步怎么做

如果要继续挖这个方向，优先级建议是：

1. 试更强的锐化
   - 更低温度
   - 或 `power > 1`
2. 同时监控 expert 是否被饿死
   - gate 熵骤降
   - shared/specific 某一路突然低秩塌缩
3. 如果更强锐化仍然无效
   - 说明问题不在 softmax“太平”
   - 而在 gate 输入特征本身没有足够判别力
