# PLE Balanced Shared Direction

## Goal

Build a new incremental direction on top of `PLE` that explicitly reduces head-domain dominance in the shared expert.

The main question is:

> Can we keep the shared expert useful for large domains while preventing it from being defined almost entirely by the largest scenarios?

This direction is separate from SHAVQ.
It should be implemented as a new PLE-based family, not as a modification of the SHAVQ models.


## Motivation

Current observations suggest:

- the shared branch is useful
- the shared branch is still biased toward the largest domains
- smaller domains can benefit from a cleaner shared prior
- but the shared prior is not yet truly domain-neutral

So the new direction is not mainly about stronger bottlenecks.
It is about **changing how the shared expert is trained**.


## Core Hypothesis

The shared expert in a multi-scenario model should not only be:

- low-capacity
- well-regularized
- or disconnected from specific branches

It should also be:

- trained with **balanced domain contribution**
- optionally constrained so that its output is **hard to identify by source domain**


## Planned Variants

### Variant A: Balanced Shared Expert

This is the first and safest incremental version.

Idea:

- keep the PLE structure
- keep the shared expert
- change the shared-side training signal so that each domain contributes more evenly

Possible implementation choices:

- per-domain reweighting of shared loss contribution
- per-domain batch aggregation before shared loss is applied
- explicit domain-balanced auxiliary loss on shared outputs

Primary question:

- Does balanced optimization make the shared expert less dominated by head domains?


### Variant B: Adversarial Shared Expert

This version adds a domain discriminator on top of shared outputs.

Idea:

- shared expert output should be informative for CTR
- but it should be hard to predict the input sample's source domain

Possible implementation choices:

- gradient reversal layer
- domain classifier attached to shared hidden
- adversarial loss that encourages domain-invariant shared representation

Primary question:

- Can we make the shared expert less domain-identifiable without harming CTR too much?

Current implementation note:

- `ple_balanced_v2` is the first adversarial version
- a raw adversarial branch on unnormalized shared hidden is unstable
- the current safe form is:
  - keep the original PLE encoder
  - apply a readout-only `LayerNorm` on shared hidden before the shared auxiliary head and domain discriminator
  - use small adversarial weight and small GRL scale

Observed early behavior on KuaiRand:

- too-strong adversarial settings can make shared readout explode numerically and collapse AUC
- mild settings can lower domain predictability from shared hidden while keeping overall AUC near baseline
- the gain pattern so far is:
  - domain0 / domain4 may improve
  - domain2 can improve under moderate balanced sharing
- overly strong pressure hurts overall and head-domain quality

`ple_balanced_v3` follows a different structural hypothesis:

- split shared information into `common` and `balanced` paths
- use a gate to let the specific tower recover information from both shared paths
- let the `common` path stay more domain-predictive while the `balanced` path gets adversarial pressure

Observed early behavior on KuaiRand:

- `ple_balanced_v3_try1` reached `overall.auc=0.786080`
- this is slightly above the current `v2` best
- the gate is not collapsed:
  - `specific/common/balanced` mean weights were roughly `0.458/0.394/0.148`
- the common path stayed more domain-identifiable than the balanced path:
  - common domain acc `0.740`
  - balanced domain acc `0.480`
- domain0 and domain4 receive noticeably more common/balanced mixing than domain1 or domain3 in the printed gate summary


### Variant C: Balanced + Adversarial

This is the later combination version.

It should only be tried after A and B are understood separately.

Reason:

- if A already solves most of the imbalance, adversarial pressure may be unnecessary
- if B is too strong, it may erase useful domain structure


## Why Start With Balanced Loss

Balanced optimization is the preferred first step because it is lower risk.

Benefits:

- easy to interpret
- easy to ablate
- less likely to over-remove useful domain structure
- directly targets the most likely root cause: sample imbalance

The adversarial version is stronger, but it can also become too aggressive.
That makes it harder to tell whether any gain comes from genuine invariance or from over-regularization.


## Structural Principle

Do not patch the current SHAVQ codebase for this direction.

Instead:

- start from PLE
- add a new balanced-sharing family
- keep the shared expert logic and the specific expert logic explicit
- treat adversarial invariance as an optional extension


## Suggested Experimental Order

### Step 1: Baseline PLE Recheck

Before changing anything, run the current PLE baseline again under the same KuaiRand setup.

This gives a fresh anchor for:

- overall AUC
- domain0 to domain4 AUC
- large-domain average
- small-domain average


### Step 2: Balanced Shared Expert

Implement the first variant with only balanced shared training.

Measure:

- does `domain1` and `domain2` stay strong?
- do `domain0`, `domain3`, `domain4` recover?
- does overall AUC improve or stay stable?


### Step 3: Domain Discriminator on Shared Output

Add a light adversarial branch only after the balanced version is understood.

Measure:

- domain prediction accuracy from shared output
- shared feature variance before and after adversarial pressure
- CTR impact on large and small domains separately


### Step 4: Combined Version

Only if the separate variants are both meaningful.


## Diagnostics to Track

### CTR Metrics

- overall AUC
- domain0 AUC
- domain1 AUC
- domain2 AUC
- domain3 AUC
- domain4 AUC
- large-domain average
- small-domain average


### Shared Representation Metrics

- shared hidden variance
- shared logit variance
- shared output absolute mean
- shared output domain separability


### Specific Representation Metrics

- specific hidden variance
- specific logit variance
- specific output absolute mean
- specific-to-shared variance ratio


### Balance / Adversarial Metrics

For balanced loss:

- per-domain loss contribution
- per-domain shared gradient magnitude

For adversarial loss:

- discriminator accuracy
- discriminator loss
- reversed-gradient effect on shared variance


## Success Criteria

Balanced sharing is useful if:

- head domains stop dominating shared learning
- small domains do not collapse
- overall AUC stays stable or improves
- shared output becomes more evenly useful across domains

Adversarial sharing is useful if:

- shared output becomes less domain-identifiable
- CTR does not degrade too much
- the balanced version alone is not sufficient


## Failure Modes to Watch

- balanced loss makes shared expert too weak
- adversarial loss removes useful domain structure
- the model becomes harder to optimize but not better
- domain1 / domain2 stay strong while others still lag
- small domains improve but overall AUC drops too much


## Handoff Notes

This direction should stay separate from SHAVQ.

Implementation order should be:

1. PLE baseline
2. balanced shared expert
3. adversarial shared expert
4. combined version

Do not start from the combined version.
Do not mix it into the current SHAVQ family.
