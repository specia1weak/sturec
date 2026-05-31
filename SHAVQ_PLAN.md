# SHAVQ Development Plan

## Goal

Build a clean ablation line for **Shared-Expert Vector Quantization (SHAVQ)** in multi-scenario recommendation.

This line is not primarily for immediate SOTA chasing. Its first objective is to answer a set of structural questions:

1. Is the current shared VQ channel too coarse because it uses only one prototype per sample?
2. Is the current specific branch carrying meaningful scenario fluctuation, or only compensating for shared branch defects?
3. When a gate or prototype combiner is introduced, does the network actually trust the weighted feature, as reflected by variance after weighting?
4. At what layer does feature death happen, if it happens at all?

The implementation strategy is:

- Create **independent model packages** under `src/betterbole/models/msr/`
- Do **not** pack all variants into one giant configurable model
- Keep each version focused on **one hypothesis**
- Keep diagnostics consistent enough to compare versions side by side


## Directory Strategy

Planned packages:

```text
src/betterbole/models/msr/shavq_v1/
src/betterbole/models/msr/shavq_v2/
src/betterbole/models/msr/shavq_v3/
src/betterbole/models/msr/shavq_v4/
```

Registration names should also stay versioned:

- `shavq_v1`
- `shavq_v2`
- `shavq_v3`
- `shavq_v4`

Principle:

- Reuse only low-level framework components such as `MSRModel`, `DomainTowerHead`, `MLP`
- Do not share the core quantizer / shared expert / prototype routing logic across SHAVQ versions
- If repeated code appears, tolerate some duplication for cleaner ablation boundaries


## Version Roadmap

## SHAVQ-V1

### Purpose

Create a clean baseline version of the current VQ-share idea with better diagnosis, not with more tricks.

### Hypothesis

The current single-prototype shared channel is viable, but the shared representation is too dominant at feature level and too coarse at semantic level.

### Core Structure

- Single spherical codebook
- Hard nearest-neighbor quantization with STE
- Residual specific branch
- Shared branch uses one prototype only
- Specific branch uses fluctuation only

### What V1 Must Answer

1. Is codebook usage healthy?
2. Is shared feature variance much larger than specific fluctuation variance?
3. Is specific logit variance still meaningful even when specific feature variance is smaller?
4. Does the model rely on shared branch too heavily at representation layer?

### Expected Signals

Healthy case:

- Code usage close to full coverage
- Code entropy high
- `quantized_cos` high
- `residual_norm` not near zero
- Shared feature variance dominates, but specific logit variance remains non-trivial

Warning case:

- Shared feature variance dominates too strongly, e.g. `feat_ratio > 0.90`
- Specific branch survives only because its head amplifies weak features


## SHAVQ-V2

### Purpose

Test whether the shared channel becomes less coarse if one sample is allowed to use a sparse combination of multiple prototypes.

### Hypothesis

The main limitation of V1 is not codebook training quality, but the fact that the shared expert sees only one prototype per sample.

### Core Structure

- Shared branch uses `Top-k` prototype routing, starting from `Top-2`
- Shared input becomes a sparse mixture of prototypes
- Specific branch remains as close as possible to V1

### What V2 Must Answer

1. Does multi-prototype shared routing reduce shared over-dominance at feature layer?
2. Does `feat_ratio` fall to a more balanced range without collapsing shared usefulness?
3. Does AUC improve when shared representation becomes finer?
4. If prototype weights are learned, does the weighted feature keep meaningful variance after weighting?

### Gate / Weight Diagnostics

If prototype weighting is introduced, record all three:

- Prototype weights mean / variance / entropy
- Raw prototype feature variance
- Weighted shared feature variance

Interpretation:

- Large raw variance but tiny weighted variance means the model distrusts that route
- Stable weighted variance means the network is actively using the route


## SHAVQ-V3

### Purpose

Test whether shared representation should be modeled as a residual composition of multiple discrete components instead of a single sparse mixture.

### Hypothesis

If V2 helps but still saturates, the problem is not just single-prototype routing but insufficient discrete compositionality.

### Core Structure

- Residual vector quantization or two-stage codebook
- Shared representation is approximated by multiple discrete components in sequence
- Specific branch still takes the residual after shared extraction

### What V3 Must Answer

1. Can multi-stage discrete sharing improve shared granularity while preserving bottleneck behavior?
2. Does residual quantization reduce pressure on specific head compensation?
3. Does shared feature variance become richer without making specific branch die?

### Expected Signals

Good case:

- `quantized_cos` remains high
- Shared feature variance grows in a controlled way
- Specific logit variance remains meaningful
- AUC improves over V1 and ideally over V2

Bad case:

- Residual branch collapses
- Shared branch monopolizes prediction
- Gate or stage-2 quantizer becomes effectively dead


## SHAVQ-V4

### Purpose

Test whether shared experts should be conditioned on prototype identity, not just prototype value.

### Hypothesis

Shared prototypes are not only compressed inputs; they may also be routing keys for different shared reasoning rules.

### Core Structure

Options:

- Prototype-conditioned shared expert
- Code-id embedding concatenated with shared vector
- Prototype-aware gate over multiple shared sub-experts

Only one of these should be implemented in V4. Do not mix them all.

### What V4 Must Answer

1. Does prototype-conditioned shared computation outperform a plain shared MLP?
2. Does conditioning increase shared expressivity without destroying the bottleneck?
3. If a gate is used, which gated branch keeps variance after weighting?


## Diagnostics Standard

All SHAVQ versions should expose a comparable diagnostic summary.

## Representation Layer Metrics

Always monitor:

- projected feature variance
- quantized shared variance
- residual variance
- shared hidden variance
- specific hidden variance
- specific fluctuation variance
- absolute mean for each important tensor

Why:

- Variance indicates expressivity and survival
- Absolute mean indicates activation magnitude


## Logit Layer Metrics

Always monitor:

- shared logit variance
- specific logit variance
- final logit variance
- shared logit absolute mean
- specific logit absolute mean
- shared logit variance ratio

Interpretation:

- Shared branch should usually dominate slightly
- Specific branch should remain non-trivial
- Near-zero specific logit variance indicates branch death


## Quantization Metrics

Always monitor:

- code usage
- code entropy
- quantized cosine similarity
- residual norm
- commitment loss

Interpretation:

- High entropy and high usage indicate healthy exploration
- High cosine plus non-zero residual means shared and specific both exist


## Gate / Weight Metrics

Whenever a gate or prototype combiner exists, always monitor:

- gate weight mean
- gate weight variance
- gate entropy
- raw branch variance before weighting
- weighted branch variance after weighting

Interpretation:

- A branch is not trusted just because its gate value is large
- The more reliable trust signal is whether weighted features keep variance


## Feature Death Criteria

Use simple operational rules to flag dead features or dead branches.

### Feature Almost Dead

Mark as near-dead if:

- variance < `1e-4`
- and abs mean < `1e-3`

for multiple windows

### Branch Almost Dead

Mark as near-dead if:

- logit variance is consistently below `20%` of the competing branch
- or weighted feature variance is near zero for multiple windows

### Pseudo-Alive Branch

Mark as pseudo-alive if:

- raw variance is not small
- but weighted variance is consistently tiny

This means the branch exists but is not trusted by the network.


## Development Order

Recommended order:

1. Implement `shavq_v1`
2. Fix diagnostic output format and naming
3. Implement `shavq_v2` with `Top-2` prototype shared routing
4. Compare V1 vs V2 before moving on
5. Implement `shavq_v3` only if V2 confirms that shared coarse granularity is the bottleneck
6. Implement `shavq_v4` only after V1 to V3 results clarify whether prototype-conditioned shared reasoning is needed


## Logging Style

Each version should print:

1. A short summary line every fixed number of steps
2. A structured recorder dump for deeper inspection

Suggested short summary pattern:

```text
[SHAVQ-vX] step=2600 code_entropy=... feat_ratio=... logit_ratio=... residual_norm=... gate_entropy=... weighted_var_ratio=...
```

The exact keys can vary by version, but the overall format should remain stable.


## Immediate Next Step

Implement `shavq_v1` first.

V1 target:

- Reproduce the current VQ-share behavior as a cleaner baseline
- Preserve the current useful diagnostics
- Make the code independent from the current `vq_share` package
- Use V1 as the reference for all later SHAVQ variants
