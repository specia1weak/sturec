# PLE Balanced V3

## Design Idea

`ple_balanced_v3` is a PLE-based incremental variant for multi-scenario recommendation.

The core motivation is:

- a single shared expert is usually pulled toward head domains
- forcing the whole shared branch to become domain-invariant can help small domains
- but that same pressure often hurts head-domain expression power

So this version no longer treats "shared" as only one thing.
It explicitly splits shared information into two paths:

- `common shared`: keeps stronger main-task expression and is allowed to remain domain-identifiable
- `balanced shared`: receives balanced supervision plus adversarial domain confusion, aiming to be more neutral across domains

The specific branch then does not work alone.
Instead, the final task tower sees a gated fusion of:

- `specific`
- `common shared`
- `balanced shared`

This means:

- head domains can still recover useful high-capacity shared patterns from `common`
- small domains can borrow cleaner cross-domain support from `balanced`
- the gate decides how much each domain should trust each source


## Key Structure

The model keeps the original `PLE` encoder as the bottom trunk, then adds:

1. Two projections on the shared hidden:
   - `common_projection`
   - `balanced_projection`
2. Readout normalization on both shared branches:
   - `LayerNorm` without affine parameters
3. Two domain-related heads:
   - `common_domain_probe`: measures how domain-specific `common` remains
   - `balanced_domain_discriminator`: adversarially suppresses domain identity in `balanced`
4. A domain-aware gate:
   - input = `[specific, common, balanced, domain_emb]`
   - output = 3 weights for `specific/common/balanced`
5. A fused final hidden:
   - `fused = ws * specific + wc * common + wb * balanced`


## What We Monitor

The debug print at eval time is designed for ablation, not just CTR:

- branch variance: `specific/common/balanced/fused`
- branch logit variance
- gate mean and gate variance
- gated feature variance after weighting
- domain prediction accuracy on `common` and `balanced`
- low-variance ratio to detect dead features
- per-domain gate allocation summary

These signals answer:

- did the two shared branches really separate?
- did the gate collapse?
- is the balanced branch actually less domain-identifiable?
- are any branches or gated features dying?


## First Result

First KuaiRand run:

- experiment: `ple_balanced_v3_try1`
- overall AUC: `0.786080`
- domain0 AUC: `0.736438`
- domain1 AUC: `0.742580`
- domain2 AUC: `0.823909`
- domain3 AUC: `0.729203`
- domain4 AUC: `0.762247`

Compared with the current `ple_balanced_v2` best (`0.785836`), this is a small but real improvement.


## Main Observations

The first run already shows the intended structural behavior:

- the gate did not collapse
- average gate weights were roughly:
  - `specific/common/balanced = 0.458 / 0.394 / 0.148`
- `common` remained much more domain-identifiable than `balanced`
  - common domain accuracy: `0.740`
  - balanced domain accuracy: `0.480`

Per-domain gate summary from the first run:

- `d0`: `0.18 / 0.59 / 0.23`
- `d1`: `0.53 / 0.34 / 0.13`
- `d2`: `0.44 / 0.45 / 0.11`
- `d3`: `0.64 / 0.27 / 0.10`
- `d4`: `0.34 / 0.48 / 0.18`

Interpretation:

- `domain0` and `domain4` are using more shared recovery
- `domain1` and `domain3` rely more on their specific path
- `balanced` is not dominant yet, but it is active and not dead


## Current Judgment

This version is promising because it improves the information flow itself, not only the loss:

- `v2` mainly cleaned the shared branch
- `v3` adds an explicit recovery path for domains that still need expressive shared information

The next tuning priorities are:

- `gate_temperature`
- `balanced_domain_adv_lambda`
- `common_probe_weight`

Those three parameters are the most likely to improve the tradeoff between:

- head-domain expressiveness
- small-domain support
- overall AUC
