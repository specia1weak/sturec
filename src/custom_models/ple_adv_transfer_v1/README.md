# PLE-Adv Transfer V1

## 目标

这个分支专门验证一个问题：

- 传统 `domain-specific expert` 只吃本域样本，导致 `domain A` 的样本无法给 `domain B` 的专属能力带来红利。
- 如果把 `specific` 分成“全域 transferable 部分”和“本域 private residual 部分”，是否能更合理地共享 `common` 表征。

## 结构

设：

- `shared_hidden`：PLE 共享分支输出
- `common_hidden = stopgrad(shared_hidden)`
- `latent_hidden`：连续投影后的补充表征
- `residual_hidden = latent_hidden - common_hidden`

则 specific 分支拆成：

- `transfer_hidden = TransferPool(common_hidden)`
- `private_hidden = PrivateDomainMLP([common_hidden, residual_hidden])`
- `specific_hidden = LN(transfer_hidden + private_hidden)`

logit 结构：

- `shared_logits = SharedHead(shared_hidden)`
- `transfer_logits = TransferHead(transfer_hidden)`
- `private_logits = DomainPrivateHead(private_hidden, domain_id)`
- `specific_logits = transfer_logits + private_logits`
- `fused_logits = shared_logits + gate * specific_logits`

## 核心思想

- `transfer_hidden` 用全域样本训练，让所有 domain 都能通过 `common` 的共享模式得到红利。
- `private_hidden` 只通过目标 domain 的头去学习剩余增量，保留本域修正能力。
- `common_hidden` 使用 `detach()`，防止 private/specific 反向污染 shared/common。

## 实验结论

### 1. Smoke

- 配置：
  - `transfer_topk=2`
  - `transfer_aux_weight=0.05`
  - `private_aux_weight=0.02`
  - `specific_domain_weight=0.01`
- 结果：
  - `overall.auc = 0.784956`
- 观察：
  - `specific_probe_acc ≈ 1.0`
  - 说明 private 仍然过度域化，transfer 没有真正接管“跨域红利”职责。

### 2. Transfer-heavy, hard top1

- 配置：
  - `transfer_topk=1`
  - `transfer_aux_weight=0.1`
  - `private_aux_weight=0.0`
  - `specific_domain_weight=0.0`
- 结果：
  - `overall.auc = 0.785330`
  - `domain4.auc = 0.758437`
- 观察：
  - `specific_probe_acc ≈ 0.011`
  - private 的 domain 可识别性被明显打掉
  - `transfer` 已经成为主要 specific 信息来源
  - 说明“全域 transfer + 弱 private”这条方向是成立的
  - 但 private 头在这个配置下几乎没有被有效利用

### 3. Transfer-heavy, soft top2

- 配置：
  - `transfer_topk=2`
  - `transfer_aux_weight=0.1`
  - `private_aux_weight=0.0`
  - `specific_domain_weight=0.0`
- 结果：
  - `overall.auc = 0.785281`
  - `domain4.auc = 0.764120`
- 观察：
  - `transfer_gate_entropy ≈ 0.341`
  - `transfer_top1_mean ≈ 0.879`
  - 比 `top1` 更软，但整体并没有超过 hard top1
  - 说明当前 transferable expert 更像“少数几个稳定模式”，不是必须做很软的组合

## 当前判断

- 这条路线是有价值的，因为它第一次把“domain A 样本能否给其他 domain 的 specific 能力带来红利”明确结构化了。
- 但当前最好的结果仍然没有超过 `ple_adv` 主线。
- 主要问题不是 transfer 不工作，而是 private 的职责还不够自然：
  - 要么太强，退化成域分类器
  - 要么太弱，几乎不工作

## 下一步

- 给 private 分支更合适的目标：
  - 学 residual margin
  - 学 counterfactual delta
  - 而不是直接再拟合一次 CTR
- 或者把 private 从“直接预测 logit”改成“只修正 transfer logit”

