# SHAVQ-V4 Notes

`shavq_v4` 是一版“共享先验 + 专属修正”的原型，结构上包含：

- 两级共享量化分支
  - `stage1` / `stage2`
- 两路专属修正分支
  - `innovation`
  - `context`
- 两组 gate
  - shared gate: 混合 `stage1` 与 `stage2`
  - specific gate: 混合 `innovation` 与 `context`

这版模型已经接入 `observatory`，重点用于做结构性排查，而不是只看最终 AUC。

## 当前重点监控对象

- `shavq_v4_shared_hidden`
- `shavq_v4_specific_feature_fluctuation`
- `shavq_v4_specific_innovation_hidden_weighted`
- `shavq_v4_specific_context_hidden_weighted`
- `shavq_v4_shared_gate_weights`
- `shavq_v4_specific_gate_weights`

训练过程中会额外打印：

- `feat_ratio`
  - 共享特征方差占比
- `logit_ratio`
  - 共享 logit 方差占比
- `delta_var`
  - innovation delta 方差
- `r2`
  - 第二阶段量化残差范数均值
- `total_cos`
  - 共享连续表示与量化表示的平均余弦相似度

## 当前已观察到的典型现象

基于 2026-05-30 的 KuaiRand 诊断日志，可以先记住下面几条：

### 1. 码本不是主要问题

- `s1=64/64`
- `s2=64/64`
- `ent1` / `ent2` 长期都比较高
- `total_cos` 会随着训练推进逐步升高

这说明两级 VQ 的覆盖率和对齐情况都还可以，当前主要瓶颈不在“码本死掉”。

### 2. shared 表征本身是健康的

`shared_hidden` 一般表现为：

- `eff_rank` 明显高于 10
- `top1_energy` 不高
- `mean_abs_corr` 不高

这意味着 shared branch 作为表示学习模块，并没有明显塌缩。

### 3. context 路容易塌成单方向

`specific_context_hidden_weighted` 是目前最值得警惕的对象，典型信号包括：

- `eff_rank` 接近 1
- `top1_energy` 接近 1
- `mean_abs_corr` 很高

这说明这一路虽然不一定输出全零，但经常会退化成“64 维里重复同一个方向”。

### 4. final specific 往往被 innovation 路主导

如果看到下面的模式长期成立：

- `specific_feature_fluctuation` 与 `specific_innovation_hidden_weighted` 的 `linear_cka` 很高
- `specific_feature_fluctuation` 与 `specific_context_hidden_weighted` 的 `linear_cka` 明显更低

那么基本可以判断：

- final specific 输出主要来自 innovation
- context 更像弱辅助项，而不是独立信息源

### 5. shared 在最终决策里的话语权可能偏低

如果：

- `feat_ratio` 还可以
- 但 `logit_ratio` 很低

说明 shared branch 学到了一些表征，但这些信息没有有效穿透到最终 CTR logits。

这通常意味着问题出在：

- head 过弱
- residual / specific 修正过强
- gate 或融合方式稀释了 shared 贡献

## 读日志时的推荐顺序

1. 看 `Recorder` 摘要行
   - 先判断码本、gate、shared/specific 方差占比、量化对齐程度
2. 看 `shared_hidden` 与 `specific_feature_fluctuation`
   - 判断两条主干谁更健康
3. 看 `specific_innovation_hidden_weighted` 与 `specific_context_hidden_weighted`
   - 判断两条专属子路谁在真正工作
4. 看 `relation_stats`
   - 判断是否出现分支冗余、假双路、或者 gate 混合无效

## 当前最值得继续优化的方向

- 提高 shared logit 侧贡献
  - shared 表征可能是健康的，但没有转化成足够的预测话语权
- 处理 context 路低秩塌缩
  - 避免专属双路退化成 innovation 单路
- 让 specific gate 学会“不平均”
  - 如果某一路已明显塌缩，gate 应能主动压低它

## 与 observatory 的配合方式

这版模型的 observatory 价值不在于多打几张图，而在于回答下面几个结构问题：

1. VQ shared 到底是健康共享，还是看起来离散、实际没有用。
2. specific 是真的双路修正，还是 innovation 一路独大。
3. gate 是真正路由，还是形式上的软平均。
4. 某条分支是“少数维度活跃”，还是“整体只剩一个主方向”。

如果后续继续演化 `shavq` 系列，建议优先保持这些诊断接口不丢。
