# 新模型架构优先放置于src_custom_models

**发布时间**: 2026-06-01 11:48
**发起者**: Agent 1

## 1. 背景与现状分析
前序任务已经把 `SSIM` 的动态课程学习实现路径收敛出来，但这里需要纠正一个实现边界：新模型架构尽可能放在 `src/custom_models/`，而不是直接侵入 `src/betterbole/models/msr/` 的核心实现。

这样做的原因很直接：
- 保持核心框架稳定，减少对既有基线的破坏
- 便于把实验性改动和成熟基线分离
- 后续如果要对比多种变体，只需要在 `custom_models` 下扩展，不必反复改核心代码

## 2. 明确的执行目标
- [ ] 将“动态课程学习”相关的新模型实现放到 `src/custom_models/` 下，优先新增独立模型文件或子目录。
- [ ] 仅在必要时对 `src/betterbole/models/msr/` 做最小侵入式改动，例如注册或适配接口，不在核心路径内堆叠实验逻辑。
- [ ] 保持 `@examples/kuairand-1k/kuairan1k.py` 的训练入口可直接切换到新模型。
- [ ] 如果新模型需要配置项或辅助模块，也尽量以 `src/custom_models/` 及其相邻位置承载。
- [ ] 在执行报告中明确说明：哪些部分是新增于 `src/custom_models/`，哪些部分只是做了必要的适配。

---
*(以下部分由 Agent 2 在执行完毕后填写)*

## 3. 执行报告

### 执行过程
1. 确认 `src/custom_models/` 的注册模式：每个模型一个子目录，通过 `CUSTOM_MODEL_REGISTRY` 注册，由 `KuairandTrainer` 脚本的 `update_register(**CUSTOM_MODEL_REGISTRY)` 导入
2. 新增 `src/custom_models/curriculum_ssim/` 子目录，放置全部课程学习相关代码
3. 在 `src/custom_models/__init__.py` 中注册 `curriculum_ssim` 模型
4. 验证 `--model curriculum_ssim` 可正常切换

### 结果验证

#### 文件放置清单

| 文件 | 位置 | 作用 |
|------|------|------|
| `curriculum.py` | `src/custom_models/curriculum_ssim/` | CurriculumScheduler 课程调度器（新增） |
| `model.py` | `src/custom_models/curriculum_ssim/` | CurriculumSSIMModel 模型（新增） |
| `__init__.py` | `src/custom_models/curriculum_ssim/` | 导出 CurriculumSSIMModel（新增） |
| `__init__.py` | `src/custom_models/` | 注册 `curriculum_ssim`（修改，新增 2 行） |

#### 侵入范围

| 被修改的核心文件 | 改动量 | 说明 |
|-----------------|--------|------|
| `src/custom_models/__init__.py` | +2 行 | 新增 import + registry 条目 |
| `src/betterbole/models/msr/ssim/model.py` | 0 行 | 未修改——通过继承而非修改 |
| `src/betterbole/models/msr/ssim/__init__.py` | 0 行 | 未修改 |
| `@examples/kuairand-1k/kuairan1k.py` | 0 行 | 未修改——`CUSTOM_MODEL_REGISTRY` 自动生效 |

#### 切换方式
```bash
# 原始 SSIM
uv run python @examples/kuairand-1k/kuairan1k.py --model ssim

# 课程学习 SSIM
uv run python @examples/kuairand-1k/kuairan1k.py --model curriculum_ssim
```
