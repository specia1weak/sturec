# PLE SHAVQ V1

## Design Idea

`ple_shavq_v1` is the first minimal fusion between the balanced PLE line and the SHAVQ idea.

It keeps the `ple_balanced_v3` trunk unchanged and only replaces the `balanced shared` branch with a VQ branch.

The intent is to test one narrow question:

- can a quantized, more discrete shared branch improve the balanced shared path without breaking the existing PLE structure?

The model keeps:

- `specific` branch
- `common shared` branch
- `balanced VQ shared` branch
- domain-aware gate over the three branches
- balanced shared auxiliary losses

The VQ branch adds:

- codebook warmup and EMA updates
- commitment loss
- optional domain-balanced codebook EMA


## What We Tested

Three useful configurations were run on KuaiRand:

1. `ple_shavq_v1` with vanilla EMA
2. `ple_shavq_v1` with domain-balanced EMA and no balanced adversarial loss
3. `ple_shavq_v1` with domain-balanced EMA, no balanced adversarial loss, and a smaller codebook

Then a fourth variant was tested:

- `ple_shavq_v2`, which adds VQ code identity to the gate input


## First Results

Best observed numbers so far:

- `ple_shavq_v1_balema_noadv`
  - overall AUC: `0.785766`
- `ple_shavq_v2_balema_noadv`
  - overall AUC: `0.785804`

For comparison:

- `ple_balanced_v3_try1`
  - overall AUC: `0.786080`

So the VQ fusion line is currently below `ple_balanced_v3`, but it is close enough to be informative.


## Main Observations

The VQ branch is not dead:

- codebook usage becomes active quickly
- the codebook is used by almost all codes
- code identity is not uniform across domains
- gate allocation changes by domain

But the branch is not yet clearly better than the dense balanced branch:

- it tends to improve `domain4` a bit
- it can reduce domain separability in the VQ branch when domain-balanced EMA is used
- the gate still often prefers `specific` and `common` over `VQ`

Adding code identity to the gate helps a little:

- gate trust in the VQ branch increases slightly
- overall AUC improves slightly over the plain V1 VQ fusion
- but it still does not surpass `ple_balanced_v3`


## Current Judgment

The evidence so far suggests:

- VQ is a meaningful representation component
- but in this setup, it is not yet a clear replacement for the balanced dense shared branch
- it looks more like a complementary robust prototype branch than a universal winner

The most likely future directions are:

- add a lightweight residual recovery path from the VQ branch
- test codebook-size sensitivity more systematically
- add a stronger code-aware routing mechanism
- inspect per-domain code usage and gate behavior more deeply

