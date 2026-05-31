# PLE SHAVQ V3

## Design Idea

`ple_shavq_v3` is the residual-recovery extension of the SHAVQ fusion line.

It builds on `ple_shavq_v2` and adds one extra idea:

- the VQ branch should not only provide a compact shared prototype
- it should also recover some information from the quantization residual

So the balanced shared path becomes:

- `balanced_latent`
- `balanced_quantized`
- `balanced_residual = balanced_latent - balanced_quantized`
- `balanced_residual_hidden = residual_projection(balanced_residual)`
- `balanced_hidden = balanced_vq_hidden + residual_scale * balanced_residual_hidden`

The rest of the structure stays the same:

- `specific` branch
- `common shared` branch
- `balanced VQ` branch
- domain-aware gate over the three branches
- optional domain-balanced codebook EMA


## Why This Version Exists

The earlier SHAVQ fusion versions showed a consistent pattern:

- VQ can form a real discrete shared prototype branch
- but the branch is not automatically better than the balanced dense shared path
- the most likely missing piece is some form of residual recovery

This version tests that directly.


## What We Observed

Three useful residual settings were run with `domain-balanced EMA` and no extra adversarial pressure:

- `residual_scale=0.5`
- `residual_scale=0.25`
- `residual_scale=0.15`

The most important result so far:

- `residual_scale=0.25` was the best among these runs

Additional codebook-size test:

- `codebook_size=48, residual_scale=0.25`
- this did not beat the best `64-code` run


## First Result Summary

Best observed run so far:

- experiment: `ple_shavq_v3_balema_noadv_r025`
- overall AUC: `0.785907`

Reference points:

- `ple_balanced_v3_try1`
  - overall AUC: `0.786080`
- `ple_shavq_v2_balema_noadv`
  - overall AUC: `0.785804`
- `ple_shavq_v1_balema_noadv`
  - overall AUC: `0.785766`


## Main Observations

The residual path is doing something real:

- the VQ branch becomes more gate-visible
- `domain4` improves notably in the stronger residual settings
- the balanced branch is no longer purely quantized, it can recover some fine-grained information

But the price is also visible:

- too much residual scale makes the branch less selective
- too little residual scale leaves the VQ branch too weak
- the overall best still sits slightly below `ple_balanced_v3`


## Current Judgment

This suggests the VQ branch is best treated as:

- a compact and robust shared prototype path
- plus a small residual recovery path

It does not yet replace the balanced dense shared branch.
It currently looks more like a complementary branch than a superior standalone shared expert.

