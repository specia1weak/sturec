# PLE-Adv DualEmb V1

这是 `ple_adv` 的一个独立实验分支，核心目标是验证：

1. `shared` 和 `specific` 是否需要不同的底层 embedding 来源。
2. 仅仅把输入切成两套 embedding table，能不能比单表更容易形成职责分工。
3. 这种分流是否会真正转化为 CTR 提升，还是只会改写几何结构。

## 结构

- `common_embedding`：只给 `shared` 分支使用。
- `joint_embedding`：给 `specific` 分支主用，同时在 `v1` 中可选择给 `shared` 提供弱辅助。
- `shared` 输入：
  - `common_x + alpha * joint_x`
- `specific` 输入：
  - `joint_x - beta * shared_x.detach()`
- 上层仍复用 `ple_adv` 的：
  - shared/specific CTR head
  - shared domain adversarial loss
  - specific domain classification loss
  - gate hardness supervision
  - counterfactual raw-logit margin

## 版本目标

### `v1`

验证“双表 + 弱耦合”是否能够让 shared/specific 的几何结构更分离，同时不明显损失整体 AUC。

### 当前发现

#### 1. `joint_to_shared_scale=0.15, specific_residual_weight=0.15`

- `overall.auc = 0.785172`
- `domain2.auc = 0.823820`
- `domain4.auc = 0.755831`
- 过程信号：
  - `shared` / `specific` 有明显分化
  - `specific_probe_acc` 接近 1
  - `shared_domain_adv_acc` 下降到更接近随机
  - 但整体 AUC 没有超过当前最好 `ple_adv` 版本

#### 2. `joint_to_shared_scale=0.0, specific_residual_weight=0.0`

- `overall.auc = 0.785038`
- `domain0.auc = 0.737427`
- 过程信号：
  - `shared` 变弱
  - `specific` 变强
  - `shared/specific` 几何相关性进一步下降
  - 但整体性能没有改善

## 目前判断

- 纯双表分流本身是有作用的，它确实会改变分支职责分配。
- 但只改 embedding source，不足以让 `specific` 真正成为 CTR 补偿器。
- `shared` 不能完全断开 `joint`，否则它会显著变弱。
- 下一步更值得试的是：
  - 缩小 `common` table 容量
  - 给 `shared` 引入更明确的职责约束
  - 让 `specific` 学更明确的 residual 或 counterfactual signal

