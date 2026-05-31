# PLE-Adv 实验记录

本文记录 `ple_adv` 的逐步消融过程，目标是把 `specific` 从“域编码器”推进到“CTR 补偿器”。

## 已验证版本

### 1. `specific` 默认模式

- 训练配置：`specific_feature_mode=specific`
- 验证结果：`overall.auc=0.785874`
- 分支结果：`auc(shared/specific/fused)=0.785625/0.623812/0.785874`
- 结论：
  - `shared` 是当前主力，几乎吃掉全部 CTR 表达。
  - `specific` 主要在做 domain probing，CTR 贡献很弱。

### 2. `latent` 仅输入

- 训练配置：`specific_feature_mode=latent`
- 验证结果：`overall.auc=0.785924`
- 分支结果：`auc(shared/specific/fused)=0.785863/0.643972/0.785924`
- 结论：
  - 只把 latent 喂给 `specific`，会让 `specific-only` 变强一些。
  - 但 fused 几乎不变，说明 `specific` 仍然没有真正影响最终 CTR 决策。
  - `shared` 和 `specific` 的表征分化更明显，说明这一步至少让两路更不相似了。

### 3. `specific_latent`

- 训练配置：`specific_feature_mode=specific_latent`
- 验证结果：`overall.auc=0.785935`
- 分支结果：`auc(shared/specific/fused)=0.785945/0.635717/0.785935`
- 结论：
  - 叠加原始 `specific` 与 latent 后，`specific-only` 没有继续提升。
  - fused 与 shared 基本持平，说明当前融合方式仍然没把 `specific` 的增量真正用起来。
  - 这一步更像是“增加了特征自由度”，不是“增加了职责约束”。

### 4. `latent_residual_only`

- 训练配置：`specific_feature_mode=latent_residual_only`
- 验证结果：`overall.auc=0.785690`
- 分支结果：`auc(shared/specific/fused)=0.785589/0.614618/0.785690`
- 结论：
  - 直接让 `specific` 学 `latent - shared.detach()`，并没有带来更好的 fused AUC。
  - 这个版本让 `specific` 更“分散”，`eff_rank` 更高，`top1_energy` 更低，但这种分散没有变成有效增益。
  - 说明 residual 本身可能有信号，但当前没有被 `specific` 的头和损失正确吸收。

### 5. `latent_residual`

- 训练配置：`specific_feature_mode=latent_residual`
- 验证结果：`overall.auc=0.785782`
- 分支结果：`auc(shared/specific/fused)=0.785621/0.631931/0.785782`
- 结论：
  - 把 residual 当成补充输入，而不是唯一输入，仍然没有带来 fused 的稳定提升。
  - `specific-only` 比 `latent_residual_only` 强，但仍然没有超过 `latent` 与 `specific_latent`。
  - 这进一步说明问题不在“是否给了 residual”，而在“specific 被训练成了什么角色”。

## 当前判断

- `shared` 仍然是主贡献分支。
- `specific` 目前更像 domain-sensitive encoder，而不是 residual compensator。
- 仅靠输入切换，不足以让 `specific` 对 CTR 形成稳定贡献。
- residual 会改变 `specific` 的几何结构，但不会自动带来更好的最终预测。
- 无监督 interaction gate 会塌缩，说明“结构上能做重加权”不等于“训练中会自动分工”。
- `unique` 更像去冗余约束，不足以单独把 gate 打开。
- `gate hardness supervision` 是当前最像 I2MoE 式 interaction 的约束，因为它把 `specific` 的启用程度和 `shared` 的样本级困难程度绑定起来了。

## 职责约束新发现

### 6. `specific_aux_ctr_loss` 有效

- 训练配置：
  - `specific_feature_mode=specific`
  - `ple_adv_specific_aux_weight=0.05`
  - `ple_adv_specific_aux_hardness_power=0.0`
- 验证结果：`overall.auc=0.785838`
- 分支结果：`auc(shared/specific/fused)=0.782687/0.774285/0.785838`
- 结论：
  - 这是目前最重要的转折点。`specific-only` AUC 被显著拉高，不再是一个弱分支。
  - 更关键的是，这一版的 `fused - shared` 在所有 domain 上都变成正值：
    - `domain0`: `+0.001741`
    - `domain1`: `+0.001245`
    - `domain2`: `+0.002994`
    - `domain3`: `+0.006785`
    - `domain4`: `+0.000084`
  - 这说明问题核心确实不是输入形式，而是缺少显式职责监督。
  - 当前不足是：`shared` 与 `specific` 的 logit 相关性仍然很高，`logit_corr=0.863`，说明 `specific` 更像“被拉起来的第二个 CTR 分支”，还不像真正的补偿器。

### 7. `specific_aux_ctr_loss + hardness`

- 训练配置：
  - `specific_feature_mode=specific`
  - `ple_adv_specific_aux_weight=0.05`
  - `ple_adv_specific_aux_hardness_power=1.0`
- 验证结果：`overall.auc=0.785850`
- 分支结果：`auc(shared/specific/fused)=0.785215/0.587747/0.785850`
- 结论：
  - hardness 加权并没有进一步改善整体表现。
  - 相比普通版，它让 `specific-only` 回落，说明当前 hardness 定义更像是在改变优化分布，而不是稳定增强补偿能力。
  - `domain2` 是最受益的域，`0.824690`，但 `domain0/1/4` 变弱了，整体上不如普通 `specific_aux_loss`。

### 8. `gated_add` without gate supervision

- 训练配置：
  - `specific_feature_mode=specific`
  - `ple_adv_specific_aux_weight=0.05`
  - `ple_adv_interaction_fusion_mode=gated_add`
- 验证结果：`overall.auc=0.785588`
- 分支结果：`auc(shared/specific/fused)=0.785551/0.783824/0.785588`
- 过程信号：
  - `specific_gate_mean=0.044`
  - `specific_gate_std=0.017`
  - `gate_shared_hard_corr≈0.08`
  - `logit_corr=0.986`
- 结论：
  - 这是一个关键负例。没有显式职责约束时，gate 会迅速塌到几乎全关。
  - `specific` 自身并不弱，但模型学到的是“别让 specific 说话”，而不是“什么时候让它补偿 shared”。

### 9. `gated_add + unique(0.01)`

- 训练配置：
  - `specific_feature_mode=specific`
  - `ple_adv_specific_aux_weight=0.05`
  - `ple_adv_interaction_fusion_mode=gated_add`
  - `ple_adv_interaction_unique_weight=0.01`
- 验证结果：`overall.auc=0.785777`
- 分支结果：`auc(shared/specific/fused)=0.785744/0.783556/0.785777`
- 过程信号：
  - `specific_gate_mean=0.036`
  - `specific_gate_std=0.015`
  - `gate_shared_hard_corr≈0.08`
  - hidden CKA(`shared`,`specific`) 降到约 `0.288`
- 结论：
  - `unique` 的确降低了 shared/specific 的几何重叠。
  - 但 gate 依旧几乎全关，说明“分离表示”本身并不会自动转化成“分配职责”。

### 10. `gated_add + gate hardness supervision`

- 训练配置：
  - `specific_feature_mode=specific`
  - `ple_adv_specific_aux_weight=0.05`
  - `ple_adv_interaction_fusion_mode=gated_add`
  - `ple_adv_interaction_gate_hardness_weight=0.05`
- 验证结果：`overall.auc=0.785888`
- 分支结果：`auc(shared/specific/fused)=0.784732/0.779549/0.785888`
- 过程信号：
  - `specific_gate_mean=0.358`
  - `specific_gate_std=0.132`
  - `gate_shared_hard_corr≈0.486`
  - `logit_corr=0.957`
- 结论：
  - 这是第一版真正把 gate 打开的方案。
  - 相比无监督 gate，它已经不再是常数门控，而是会对 shared 的困难样本做出反应。
  - 这版也是当前 interaction 方向里最有说服力的结果，因为它第一次给出了“模型确实在样本级分工”的过程证据。

### 11. `gated_add + gate hardness supervision + unique(0.01)`

- 训练配置：
  - `specific_feature_mode=specific`
  - `ple_adv_specific_aux_weight=0.05`
  - `ple_adv_interaction_fusion_mode=gated_add`
  - `ple_adv_interaction_gate_hardness_weight=0.05`
  - `ple_adv_interaction_unique_weight=0.01`
- 验证结果：`overall.auc=0.785802`
- 分支结果：`auc(shared/specific/fused)=0.784509/0.780461/0.785802`
- 过程信号：
  - `specific_gate_mean=0.358`
  - `specific_gate_std=0.132`
  - `gate_shared_hard_corr≈0.485`
  - hidden CKA(`shared`,`specific`) 进一步降到约 `0.268`
- 结论：
  - `unique` 和 gate hardness 在过程层面是互补的。
  - 但从当前结果看，真正决定 gate 是否工作的是 `gate hardness supervision`，不是 `unique`。
  - `unique` 更像辅助项，用来压低冗余，而不是决定性职责约束。

### 12. `gated_add + gate hardness + counterfactual raw-logit margin`

- 训练配置：
  - `specific_feature_mode=specific`
  - `ple_adv_specific_aux_weight=0.05`
  - `ple_adv_interaction_fusion_mode=gated_add`
  - `ple_adv_interaction_gate_hardness_weight=0.05`
  - `ple_adv_counterfactual_logit_margin_weight=0.01`
  - `ple_adv_counterfactual_logit_margin=0.1`
  - `ple_adv_counterfactual_noise_std=0.05`
- 验证结果：`overall.auc=0.785843`
- 分支结果：`auc(shared/specific/fused)=0.784296/0.778896/0.785843`
- 过程信号：
  - `cf_margin_loss≈0.039`
  - `counterfactual_gap_mean≈0.039`
  - `counterfactual_gate_gap_mean≈0.005`
  - `specific_gate_mean≈0.364`
  - `gate_shared_hard_corr≈0.484`
  - `logit_corr=0.954`
- 结论：
  - 这个版本说明 raw CTR logits 上的反事实 margin 是可训练的，且确实让“正常前向”和“噪声反事实前向”之间形成了稳定差距。
  - 它没有明显提升 AUC，但把 `shared/specific` 的 logit 相关性进一步压低了，说明它更像是在补充分工信号，而不是直接做性能增益。
  - 当前看，这个 loss 值得保留作为过程约束和诊断工具，后续更可能需要继续调 `margin`、`noise_std`，以及考虑把噪声替换成更强的 domain-aware swap。

## 下一步计划

1. 继续沿着职责约束做实验，不再优先改输入形式。
2. 下一步优先考虑：
   - 让 `specific` 只对 shared 难样本承担更大 loss
   - 或让 `specific` 明确拟合 `shared` 未解释掉的部分，而不是继续并行拟合 label
3. 如果继续掉点，再回头做 feature 分析，检查：
   - 特征方差是否集中在少数维度
   - `specific` 是否继续高可识别 domain
   - `specific` 对 domain2/domain4 这类小场景是否有真实增益
4. 当前阶段结论：单纯改 `specific` 输入，不足以解决分工问题。
5. 当前更靠谱的路线是 `specific_aux_loss`，而不是 hardness 版；如果继续调，优先在 `specific_aux_weight` 上做细粒度扫描。
6. interaction 方向上，当前优先级已经很明确：
   - 先扫 `ple_adv_interaction_gate_hardness_weight`
   - 再看 `interaction_unique_weight` 是否只在少量范围内提供辅助增益
7. 反事实方向上，raw-logit margin 已经证明“能制造可观测 gap”，下一步优先级是：
   - 先调 `ple_adv_counterfactual_logit_margin` 与 `ple_adv_counterfactual_noise_std`
   - 再把纯噪声替换成更贴业务语义的 `cross-domain specific swap`

## DualEmb 新分支

### 13. `ple_adv_dualemb_v1` / 双 embedding 表

- 训练配置：
  - `common_embedding` + `joint_embedding`
  - `shared` 默认看 `common_x + 0.15 * joint_x`
  - `specific` 默认看 `joint_x - 0.15 * shared_x.detach()`
  - 其余损失沿用 `ple_adv`
- 验证结果：
  - `overall.auc=0.785172`
  - `domain0.auc=0.736013`
  - `domain1.auc=0.742223`
  - `domain2.auc=0.823820`
  - `domain3.auc=0.728246`
  - `domain4.auc=0.755831`
- 过程信号：
  - `shared` / `specific` 明显分化
  - `specific_probe_acc≈0.998`
  - `shared_domain_adv_acc≈0.336`
  - `shared/specific` hidden CKA 降到很低
- 结论：
  - 双表确实让职责边界更清楚，但这版并没有超过现有最好 `ple_adv`。
  - `shared` 不能完全不看 `joint`，否则它会明显退化。
  - 仅靠底层双表，不足以解决 `specific` 的职责问题，下一步更值得做的是“容量不对称”而不是“对称双表”。

### 14. `ple_adv_dualemb_v1` / 纯分表

- 训练配置：
  - `joint_to_shared_scale=0.0`
  - `specific_residual_weight=0.0`
- 验证结果：
  - `overall.auc=0.785038`
  - `domain0.auc=0.737427`
  - `domain1.auc=0.742170`
  - `domain2.auc=0.820179`
  - `domain3.auc=0.728547`
  - `domain4.auc=0.752228`
- 过程信号：
  - `shared` 变弱，`specific` 变强
  - `shared/specific` 几何相关性进一步下降
  - `specific_probe_acc` 仍然接近 1
- 结论：
  - 纯分表会更“干净”，但牺牲了 `shared` 对整体 CTR 的支撑。
  - 它证明了分流有效，但也证明了分流不是最终答案。

## Transfer 新分支

### 15. `ple_adv_transfer_v1` / transferable + private residual

- 结构：
  - `common = stopgrad(shared_hidden)`
  - `transfer = TransferPool(common)`
  - `private = DomainPrivate([common, latent - common])`
  - `specific = transfer + private`
- 目的：
  - 让全域样本都能更新 `transfer`
  - 只让目标域样本通过 private 头做额外修正

### 16. Smoke

- 配置：
  - `transfer_topk=2`
  - `transfer_aux_weight=0.05`
  - `private_aux_weight=0.02`
  - `specific_domain_weight=0.01`
- 结果：
  - `overall.auc=0.784956`
- 过程信号：
  - `specific_probe_acc≈1.0`
- 结论：
  - 这个版本的 private 还是太域化，transfer 的共享红利没有真正接过 specific 主导权。

### 17. Transfer-heavy / hard top1

- 配置：
  - `transfer_topk=1`
  - `transfer_aux_weight=0.1`
  - `private_aux_weight=0.0`
  - `specific_domain_weight=0.0`
- 结果：
  - `overall.auc=0.785330`
  - `domain4.auc=0.758437`
- 过程信号：
  - `specific_probe_acc≈0.011`
  - `transfer_gate_entropy≈0.0`
  - `transfer_top1_mean≈1.0`
- 结论：
  - 一旦去掉对 private 的域化强化，specific 的红利确实会更偏向 transfer。
  - 这说明“让其他 expert 通过 common 吃到全域样本红利”这件事是能做到的。
  - 但代价是 private 角色变得很弱。

### 18. Transfer-heavy / soft top2

- 配置：
  - `transfer_topk=2`
  - `transfer_aux_weight=0.1`
  - `private_aux_weight=0.0`
  - `specific_domain_weight=0.0`
- 结果：
  - `overall.auc=0.785281`
  - `domain4.auc=0.764120`
- 过程信号：
  - `transfer_gate_entropy≈0.341`
  - `transfer_top1_mean≈0.879`
- 结论：
  - 更软的 transfer 组合没有带来更高 overall，但进一步抬高了 `domain4`。
  - 说明 transferable expert 这条线对小域更友好，但还需要更强的 private 设计去稳住整体表现。
