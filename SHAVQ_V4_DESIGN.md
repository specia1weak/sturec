# SHAVQ-V4 Design Note

## Purpose

This document summarizes the current **SHAVQ-V4** design in a form suitable for handoff and further optimization.

The main goal of V4 is:

- keep the strong **cross-scenario information support** that helps smaller domains
- reduce the weakness of earlier SHAVQ variants on larger domains
- explicitly separate:
  - shared prior modeling
  - innovation modeling
  - context-conditioned correction


## High-Level Idea

Compared with earlier versions:

- `v1`: one shared code, one specific residual
- `v2`: sparse prototype routing, but shared branch could become too strong
- `v3`: residual quantization was structurally clean, but specific branch was too starved
- `v3b`: specific branch improved after seeing detached shared context
- `v3c`: shared branch became more interpretable after splitting coarse/fine shared experts, but gains did not fully convert into AUC

`v4` integrates the lessons above:

1. Shared branch and specific branch do **not** use the same latent directly.
2. Shared branch models a **hierarchical discrete prior**.
3. Specific branch has **two different input channels**, not only one residual source.
4. Specific correction is **confidence-modulated**, rather than always added at full strength.


## Core Tensor Flow

Let the raw bottom feature be `x`.

### 1. Dual Encoders

Two separate encoders are used:

- `z_shared = shared_encoder(x)`
- `z_specific = specific_encoder(x)`

Meaning:

- `z_shared` is optimized to enter the discrete shared bottleneck
- `z_specific` is optimized to carry scenario-sensitive continuous information

This avoids forcing the specific branch to live only inside the leftover geometry of the shared branch.


### 2. Hierarchical Shared Quantization

The shared branch uses two-stage residual quantization:

- `quantized_stage1 = codebook_stage1(z_shared)`
- `residual1 = z_shared - quantized_stage1`
- `quantized_stage2 = codebook_stage2(residual1)`
- `shared_quantized = quantized_stage1 + quantized_stage2`
- `residual2 = z_shared - shared_quantized`

Design intention:

- stage 1 captures coarse platform consensus
- stage 2 captures finer shared correction
- the remaining residual should be what shared prior still cannot explain

Codebooks are still updated by EMA, not by BCE gradients.


### 3. Shared Expert Design

The shared branch does not use a single expert any more.

Instead:

- `shared_stage1_expert` consumes:
  - `quantized_stage1`
  - stage-1 code embedding

- `shared_stage2_expert` consumes:
  - `quantized_stage2`
  - stage-2 code embedding
  - `residual1_norm`

Then a learned gate fuses the two:

- `shared_gate_input = [quantized_stage1, quantized_stage2, stage1_code_embed, stage2_code_embed, residual stats]`
- `shared_gate_weights = softmax(shared_gate(shared_gate_input))`
- `shared_hidden = w1 * hidden_stage1 + w2 * hidden_stage2`

Shared logits are also decomposed:

- `shared_stage1_logits_raw`
- `shared_stage2_logits_raw`
- `shared_logits = w1 * logit1 + w2 * logit2`

This makes it possible to inspect:

- whether the model trusts coarse shared reasoning or fine shared reasoning
- whether stage-2 is actually used after weighting


## Specific Expert Design

This is the most important change in V4.

Earlier SHAVQ variants mostly assumed:

- specific branch input = residual, or residual + detached shared context

V4 assumes that scenario-specific modeling should have **multiple information sources**.

### 1. Shared Prior Aligned into Specific Space

The quantized shared contexts are projected into the specific latent space:

- `aligned_stage1 = linear(quantized_stage1)`
- `aligned_stage2 = linear(quantized_stage2)`
- `shared_prior_specific = aligned_stage1 + aligned_stage2`

Then define innovation:

- `innovation_delta = z_specific - shared_prior_specific`

Meaning:

- `z_specific` says what this sample wants to express in a continuous scenario-sensitive space
- `shared_prior_specific` says what the shared prior thinks it should look like
- `innovation_delta` is the difference between them


### 2. Two Specific Channels

#### A. Innovation Channel

The innovation expert consumes:

- scaled `z_specific`
- scaled `innovation_delta`
- relation features

This branch tries to answer:

- what new scenario-specific signal exists beyond the shared prior

#### B. Context Channel

The context expert consumes:

- aligned stage-1 shared context
- aligned stage-2 shared context
- relation features
- domain context embedding

This branch tries to answer:

- how the shared prior should be interpreted under the current scenario


### 3. Relation Features

The two specific channels are not fed only raw latent vectors.

They also receive relation statistics such as:

- `residual1_norm`
- `residual2_norm`
- `total_cos(shared_quantized, z_shared)`
- cosine between `z_specific` and aligned stage-1 prior
- cosine between `z_specific` and aligned stage-2 prior
- shared gate weights

These features provide explicit evidence about:

- how well the shared branch has already explained the sample
- how much unexplained innovation remains


### 4. Specific Gate

The model does not assume both specific channels are equally useful.

Instead:

- `specific_gate_weights = softmax(specific_gate(relation_features + domain_context))`
- `specific_hidden = a * innovation_hidden + b * context_hidden`

This gate answers:

- when should the model rely more on innovation
- when should the model rely more on context-conditioned interpretation


### 5. Confidence-Modulated Specific Correction

The final specific branch is not always added with full strength.

Instead:

- `specific_logits_base = tower(fluctuation(specific_hidden))`
- `specific_confidence = sigmoid(confidence_head(relation_features + domain_context))`
- `specific_logits = specific_confidence * specific_logits_base`
- `final_logits = shared_logits + specific_logits`

This is intentional.

Not every sample needs strong scenario correction:

- if shared prior already explains the sample well, specific correction should be restrained
- if innovation is large, specific correction should intervene more strongly


## Why `FeatureBifurcator` Is Used

The specific branch still uses `FeatureBifurcator`, and generally only the `Fluctuation` part is used for prediction.

Interpretation:

- the specific branch is expected to model dynamic scenario-sensitive deviation
- feature fluctuation magnitude and variance are treated as meaningful indicators of feature importance

This is consistent with the earlier SHAVQ analysis principle.


## Diagnostics That V4 Exposes

V4 records richer diagnostics than earlier versions.

### Shared Side

- stage-1 / stage-2 code usage
- stage-1 / stage-2 entropy
- stage-1 / stage-2 cosine
- stage-1 / stage-2 hidden variance
- stage-1 / stage-2 weighted hidden variance
- shared gate weights mean / variance / entropy
- shared logits raw / weighted variance

### Specific Side

- `z_specific` variance
- `innovation_delta` variance
- aligned shared prior variance
- innovation expert hidden variance
- context expert hidden variance
- weighted innovation/context hidden variance
- specific gate weights mean / variance / entropy
- specific confidence mean / variance
- specific logits before confidence
- specific logits after confidence
- fluctuation feature variance

### Joint Signals

- `feature_var_shared_ratio`
- `logit_var_shared_ratio`
- `residual1_norm_mean`
- `residual2_norm_mean`
- `total_cos_mean`


## Current Empirical Reading

At the current default settings, V4 behaves like this:

- more balanced than earlier SHAVQ variants at feature level
- shared and specific branches both remain alive
- specific correction is no longer a naive always-on additive term
- small-domain friendliness is still preserved to a meaningful extent

But the current weaknesses are also clear:

- large-domain modeling is still not as strong as the best `Crocodile-v1`
- specific confidence can become conservative
- stage-2 aligned prior in specific space may still be too weak
- the best `v3b` point is still a slightly stronger metric point than current default `v4`

In short:

- V4 is a stronger structural framework
- it is not yet the strongest tuned point


## Why V4 Matters Even If It Is Not Yet Best

V4 is important because it converts previous ad hoc conclusions into explicit modules:

- detached shared context helps specific branch
- one specific input source is not enough
- coarse shared and fine shared should be separated
- gating trust should be measured after weighting
- specific correction should be confidence-aware

This makes future optimization easier and more interpretable.


## Recommended Optimization Directions

These are the most promising next steps.

### 1. Strengthen Large-Domain Capacity

Current SHAVQ line is relatively friendly to smaller domains, but still weaker than the strongest large-capacity baseline on larger domains.

Possible directions:

- allow domain-aware modulation inside the shared experts
- allow large domains to preserve more private capacity before correction
- increase capacity of the specific context channel without letting it poison the shared codebook


### 2. Improve Specific Confidence Behavior

Current confidence can become too cautious.

Try:

- better initialization
- temperature or scale control
- confidence supervision from residual statistics
- per-domain confidence calibration


### 3. Strengthen Stage-2 Specific Utility

The stage-2 aligned prior may be too weak after mapping into specific space.

Try:

- stronger `stage2 -> specific` adapter
- nonlinear adapter instead of single linear projection
- explicit normalization or rescaling before fusion


### 4. Hybridize with Larger Bottom Capacity

`Crocodile-v1` suggests that stronger bottom-level or scenario-level capacity still matters a lot for large domains.

A practical direction is:

- keep SHAVQ-style shared prior and correction logic
- combine it with a stronger bottom or richer per-domain table mechanism


## Files

Main implementation:

- `src/betterbole/models/msr/shavq_v4/model.py`

Example entry:

- `@examples/kuairand-1k/kuairan1k.py`

Registry:

- `src/betterbole/models/msr/__init__.py`


## One-Sentence Summary

`SHAVQ-V4` is a **hierarchical shared-prior + innovation-correction** architecture:

- shared branch learns discrete cross-scenario consensus in two stages
- specific branch learns both continuous innovation and context-conditioned correction
- final scenario-specific adjustment is applied with learned confidence instead of naive full-strength addition
