# Ideas to improve FlashHead

Directions worth exploring, grouped by what part of the method they attack. Written for a
junior engineer. Background on the two stages lives in `THESIS_ADAPTIVE_PROBING.md`; the
clustering internals in `DIFFERENCE_OURS_VS_FLASHEAD.md`.

Quick reminder of the cost structure:
- **Stage 1 (routing):** score all `K` centroids, `sims = C·h`, take the top `P`. Cost does
  not depend on `P`. This is the floor on head speedup.
- **Stage 2 (refine):** gather the `P·cap` tokens in those clusters, score them exactly, pick
  the best. Cost grows with `P`.

Most prior work here (adaptive probing) shrinks stage 2. The most under-explored lever is
stage 1.

## Attacks the stage-1 floor

### 1. Hierarchical / coarse-to-fine routing

Today stage 1 is one flat matmul over all `K` centroids (`K=9496` for Qwen3-0.6B). That cost
grows as you add clusters, and the paper scales `K` into the tens of thousands. Cluster the
centroids themselves into about `√K` super-centroids, then route in two cheap steps: score the
super-centroids, descend into the winners, score only the leaf centroids underneath. Stage 1
drops from `O(K·D)` to roughly `O(√K·D)`.

This is the inverted-multi-index idea applied one level up. It is the only lever that lowers
the floor that adaptive probing cannot touch, and it gets more valuable exactly where the
paper is heading (large `K`).

**POC result: tested, not worth it in the naive form.** See `logs/hierarchical_stage1_poc.py`.
On Qwen3-0.6B (4000 WikiText-2 positions, flat baseline agreement at P=256 is 96.8%):

- Single-assignment 2-level routing: M=64, m=8 cuts stage 1 by ~13x but agreement drops to
  59%.
- Soft assignment (each leaf belongs to its top-`r` super-clusters) helps but not enough. The
  best operating point, M=100, r=3, m=16, gets 4.6x reduction at 75% agreement.

Two findings explain why:

1. **Recall is the wall.** Across every setting `agree@256 ≈ reachable`: once the true
   cluster's super-cluster is in the top-`m`, the leaf ranking succeeds. The coarse prune
   itself drops the true cluster. Getting reachability near 100% needs probing so many
   super-clusters that the savings vanish.
2. **The heavy tail that makes adaptive probing good is what breaks this.** The true cluster
   often sits deep in the flat order (p90=80, p99=1165, from the adaptive-probing analysis). A
   coarse first level cannot keep a flat-rank-500 cluster in its top few super-clusters, so it
   prunes exactly the hard tokens.

Add to that: stage 1 on Qwen3-0.6B is about 0.35 ms (1.6% of a decode step), so even a working
30x reduction would be near-invisible end-to-end. Hierarchical stage-1 only pays off at very
large `K`, and only if the coarse level is made nearly lossless (a learned coarse quantizer,
or super-centroids fit to maximize reachability instead of cosine k-means). High bar, small
payoff. Parked.

### 2. Product-quantized centroids for stage 1

The stage-1 matmul is memory-bound on the `K·D` centroid matrix. Product-quantize or low-rank
the centroids so the routing read shrinks. Smaller win than hierarchical, but it stacks with
it and with the int4 stage-1 already in use.

## Cheap to test, could lower required-P directly

### 3. MIPS-aware cluster ranking (a one-array swap)

Stage 1 ranks clusters by cosine (normalized centroid · h). The actual target is the cluster
containing `argmax(e·h)`, a raw inner product. Normalizing the centroid throws away
embedding-norm information, and in maximum-inner-product search that norm is what signals a
cluster might hold a high-scoring token. Test ranking by the unnormalized mean embedding, or
by the mean plus a per-cluster norm or max-member correction term. Same shape as the
data-aware routing POC: swap the routing matrix, recompute required-P on the same hidden
states. Fast to falsify, grounded in standard MIPS theory.

**POC result: tested, no win.** See `logs/mips_routing_poc.py`. Same 4000 WikiText-2
positions, cosine baseline top-1 agreement at P=256 is 96.8%. Three routing swaps, groups held
fixed:

| routing | p50 | p99 | mean rank | agree@256 |
|---|---|---|---|---|
| cosine (baseline) | 5 | 1165 | 75.7 | **96.8%** |
| raw mean (`mu·h`, unnormalized) | 4 | 1110 | 71.8 | 95.8% |
| `mu·h + ‖h‖·radius` (MIPS bound) | 4268 | 8824 | 4107 | 13.2% |
| `mu·h + ‖h‖·maxnorm` | 4005 | 9177 | 4110 | 11.6% |

The norm-bound variants are catastrophic: the radius term swamps `mu·h`, so they sort clusters
by radius and ignore `h`. A small-coefficient sweep (`mu·h + a·‖h‖·radius`) finds a marginal
tail gain at `a=0.01` (p99 1165→977, mean 75.7→68.6) but agree@256 still tops out at 96.2%,
under the cosine baseline.

Why cosine wins: the equal-cap balanced clusters have similar radii, so the norm/radius MIPS
signal carries little discriminative information and only adds variance. The embeddings sit
near a shared norm scale, so cosine ≈ inner product for ranking, minus the harmful norm noise.
Inner-product framing nudges the median (p50 4 vs 5) but loses the metric that matters (top-1).
Parked.

## Improves fidelity, not speed

### 4. Coverage-corrected probabilities

A true token in an unprobed cluster gets about zero probability, which caps perplexity and
distorts sampling. Each stage-1 centroid logit is a proxy for its cluster's total mass. Add an
analytic tail correction from the unprobed centroids' logits so the full distribution stays
calibrated without scoring all `V`. This makes likelihood and sampling first-class instead of
needing the Monte-Carlo workaround the paper uses.

**POC result: tested, clear win on the part that matters.** See `logs/coverage_correction_poc.py`.
Qwen3-0.6B, 1000 WikiText-2 positions, genuine corpus next-token as the target (not the model's
argmax), P=256. At P=256, 11.2% of real next-tokens land in an unprobed cluster, so the hole is
material.

| method | PPL | unprobed-only PPL |
|---|---|---|
| gold (full-V softmax) | 11.1 | 1640 |
| truncated (current FlashHead) | **10277** | ~1e30 |
| corrected Z + cluster-mean numerator (fully deployable, token unknown) | 33.1 | 5.6e7 |
| corrected Z + exact single-token logit (token known) | **9.76** | 1026 |

Two findings:

1. **The denominator fix is the real, cheap, robust win.** Add `cap · Σ exp(mean-logit_k)` over
   unprobed clusters to the softmax `Z`. The mean-logit is `mu_k · h` (raw mean embedding), one
   extra K-vector gemv, or store a per-cluster norm and reuse the stage-1 cosine scores. With
   that `Z` and the true token's *exact* logit, PPL is 9.76, matching gold's 11.1. The current
   method's 10277 was unusable for likelihood; this makes likelihood first-class.

2. **Estimating an *unknown* unprobed token's own logit from cluster aggregates fails.** Raw mean
   (33.1) and cosine-times-typical-norm (33.0) both underestimate by ~11 nats: a corpus-selected
   token sits far above any cluster average. A single global offset matches the PPL *number* but
   only as a mean-matching artifact (per-token probabilities stay biased). So for free-running
   sampling, use the corrected `Z` (fixes the over-confidence on probed tokens) and sample the
   unprobed tail by cluster mean-mass; do not trust individual unprobed token probabilities.

**Where this pays off directly:** any path where the token is known — PPL/likelihood evaluation,
and the speculative-decoding acceptance test (#5 below needs `P(drafted token)` for specific
tokens). Score those exact logits and add the corrected `Z`: calibrated, gold-quality, no full-V
softmax. This unblocks the spec-decode composition.

**Overhead only applies when the graph emits probabilities** (sampling, PPL, spec-decode
acceptance). Greedy argmax decoding — the fused default — never builds a softmax, so the tail
term is never computed and the correction costs nothing there. When probabilities are needed,
the added work is the unprobed-cluster tail (`Σ exp(mean-logit)` over `K`), which is one
extra stage-1-sized gemv (`mu·h`) plus an `O(K)` exp-and-reduce, or just the `O(K)` exp-and-
reduce if the per-cluster norms are stored and the stage-1 cosine scores are reused.

## Concrete realization of adaptive probing

### 5. Cascade probing

Instead of predicting `P` up front, probe a small `P` first, check the confidence gap between
the top two refined logits, and re-probe with more clusters only when the gap is small. Stage
1 already ran, so the second pass is just extra gathers. This sidesteps the "can we predict
required-P" question by measuring uncertainty after a cheap first look. A low-risk variant of
the adaptive-probing thesis.

## Not worth it

- **Variable cluster size by token importance.** Fights the equal-cap kernel requirement, and
  the paper's own ablation (their Table 6) says equal clusters beat unequal.
- **Caching cluster rankings across decode steps.** Hidden states move too much per token for
  temporal reuse to pay off.

## Suggested order

1. Cascade probing (low-risk variant of adaptive probing). Untested.

Done, positive: coverage correction (#4) — corrected-`Z` likelihood matches gold PPL (10277 →
9.76) when the token is known; unblocks PPL eval and the spec-decode acceptance test. Promote
to implementation: store per-cluster mean-logit norms in the npz, add the tail term where the
graph emits probabilities.

Parked: MIPS-aware ranking (#3) — POC showed cosine already beats every inner-product /
norm-bound routing on top-1. Hierarchical stage-1 (#1) — POC showed it is recall-bound and
only relevant at large `K`. Product-quantized centroids (#2) — small win, only useful stacked
on a working #1.
