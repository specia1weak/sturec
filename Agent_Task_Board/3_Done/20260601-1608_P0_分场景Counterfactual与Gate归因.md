# 分场景 Counterfactual 与 Gate 归因

**发布时间**: 2026-06-01 16:08
**发起者**: Agent 1

## 1. 背景与现状分析
上一轮已经确认 `shared VQ` 本身有真实价值，但“为什么有些场景上涨、有些场景掉点”还不够清楚。

一个可能错误的直觉是：所有小场景都应该被 `shared` 保护，`specific` 只会带来噪声。

这轮的目标就是直接做可证伪实验，不靠猜：

- 统计各 `domain` 的 `shared/specific` 实际使用强度
- 做不重训的 counterfactual：只把某个 `domain` 切回 `shared_only`，看 AUC 会不会立刻变好
- 对比 `vq_share_gate_v1` 与 plain `vq_share`，判断问题是 gate 造成的，还是 base residual 本来就偏了

## 2. 明确的执行目标
- [ ] 为 `vq_branch_probe.py` 增加按 `domain` 的 gate / residual / logit 使用统计
- [ ] 增加 `full_except_domainX_shared` 系列 counterfactual 评估
- [ ] 重新校准 `domain0-4` 与 `tab` 的真实映射，避免误读场景编号
- [ ] 根据新证据修正下一步 `vq_share_gate_v2` 的研究重点

---
*(以下部分由 Agent 2 在执行完毕后填写)*

## 3. 执行报告 (移入 3_Done 时填写)
- **执行过程**:
  1. 修正 `tab` 与 `domain` 的解释方式，确认当前工程的内部编码顺序是字符串顺序：
     - `domain0 -> tab0`
     - `domain1 -> tab1`
     - `domain2 -> tab2`
     - `domain3 -> tab4`
     - `domain4 -> tab6`
  2. 在 `@examples/kuairand-1k/vq_branch_probe.py` 中新增：
     - `tab_to_domain_vocab_order`
     - `domain_to_tab_vocab_order`
     - `domain_debug_summary`
     - `full_except_domainX_shared` 系列 counterfactual
  3. 运行：
     - `uv run python @examples/kuairand-1k/vq_branch_probe.py --model vq_share_gate_v1 --seed 2024 --max_epochs 2 --device cuda --log_name vq_branch_probe_gate_v1_cf_seed2024.log --report_name vq_branch_probe_gate_v1_cf_seed2024.json`
     - `uv run python @examples/kuairand-1k/vq_branch_probe.py --model vq_share --seed 2024 --max_epochs 2 --device cuda --log_name vq_branch_probe_vq_share_cf_seed2024.log --report_name vq_branch_probe_vq_share_cf_seed2024.json`
- **结果验证**:
  1. `domain2` 和 `domain4` 不是同一种问题。
     - `vq_share_gate_v1` epoch2：
       - `domain2(tab2)`: `full=0.822172`, `shared_only=0.809998`，说明它虽然样本少，但依然受益于 `specific`
       - `domain4(tab6)`: `full=0.743034`, `shared_only=0.761231`，说明它被 `specific` 明确伤害
     - counterfactual 也一致：
       - `full_except_domain2_shared` 的 overall 只比 full 低 `0.000722`
       - `full_except_domain4_shared` 的 overall 比 full 低 `0.002672`，但 `domain4` 自己涨了 `0.018197`
     - 结论：不能再把“所有低频 domain 都应该少用 specific”当成统一策略。`domain2` 需要 residual，`domain4` 才是明显的 over-correction 受害者。
  2. `domain4(tab6)` 的问题在 plain `vq_share` 中就已经存在，`gate_v1` 不是凭空制造了它。
     - `vq_share` epoch2：
       - `domain4 full=0.749367`
       - `domain4 shared_only=0.759463`
       - `domain4` 的 `specific_to_shared_abs_ratio = 1.579`
     - `vq_share_gate_v1` epoch2：
       - `domain4 full=0.743034`
       - `domain4 shared_only=0.761231`
       - `domain4` 的 `specific_to_shared_abs_ratio = 1.663`
       - `gate_specific_mean = 0.738`
       - `residual_gain_mean = 1.390`
     - 结论：
       - base residual 在 `domain4` 上本来就偏大
       - `gate_v1` 又进一步把这类样本路由到更高的 `specific` 权重上，所以 `domain4` 受伤更明显
  3. 哪些场景真的依赖 residual：
     - `vq_share_gate_v1` epoch2：
       - `domain0(tab0)`: `specific_to_shared_abs_ratio = 1.476`，但这是“有用的大修正”，full 相比 shared 提升 `0.104624`
       - `domain1(tab1)`: `specific_to_shared_abs_ratio = 0.251`，主要还是 shared 主导
       - `domain2(tab2)`: `specific_to_shared_abs_ratio = 0.223`，specific 较小但有效
       - `domain3(tab4)`: `specific_to_shared_abs_ratio = 0.597`，full 明显好于 shared
       - `domain4(tab6)`: `specific_to_shared_abs_ratio = 1.663`，且是“错误的大修正”
     - 这说明问题不是“specific 越大越差”，而是“某些 domain 的 specific 被放大后方向是错的”。
  4. 对下一步结构的直接约束：
     - 不应再做“统一压低所有小场景的 specific”。
     - 更值得做的是：
       - 只对 `domain4(tab6)` 这类高风险场景做 shared 保护
       - 或者让 gate 学会识别“高 residual_norm / 低 quantized_cos / 低样本 domain”时减少修正
       - 同时保留 `domain0(tab0)` 与 `domain2(tab2)` 的有效 residual 通路

## 4. 异常与反问 (移入 4_Block 时填写)
- **阻碍原因**: 无
- **需要澄清的问题**: 无
