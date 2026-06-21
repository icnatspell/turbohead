# learned_metric

**Status: PARKED — negative result. A discriminatively-trained linear routing metric improves the
median but wrecks the tail, so top-1 agreement falls. Closes the last open cell in the routing-matrix
line: cosine-on-the-embedding-mean is at the ceiling for ANY single linear routing pass, learned or
fixed.**

## Objective

Lift top-1 agreement (= stage-1 recall) at **zero inference cost** by learning a better routing
*score*, keeping the same `K x D` gemv. Agreement = the fraction of positions whose true cluster
lands in the top-P by routing score (stage 2 is exact), so a sharper routing order lifts agreement
for free — if it stays one linear pass that folds into the centroids.

## The idea (read this if you're new)

Stage-1 routes by cosine `h . ĉ_k` against each cluster's mean-embedding direction. Prior POCs swapped
that score for *fixed, unsupervised* choices and all lost:

- `recall_lift`: mean-centering, diagonal whitening, per-cluster learned-`h` prototype.
- `whitened_routing`: the full Mahalanobis metric `A = (Σ + λI)⁻¹` — loses monotonically; LLM
  hidden-state anisotropy is **signal**, and whitening deletes it.
- `factorized_router`: PQ-approximated routing — loses; recall is razor-sensitive at the top-P cut.
- A pure **orthogonal rotation** is a *provable no-op* for single-codebook cosine routing
  (`cos(Rh, Rμ) = cos(h, μ)`); OPQ-style rotation only bites paired with a product quantizer.

What none of them did: **train** the metric on the recall objective itself. ScaNN's real contribution
(Guo et al., ICML 2020) is a *score-aware* loss; `anisotropic_clustering` applied it to the
**partition** (the graduated `--eta` knob). This POC applies the discriminative version to the
**query metric**: learn a general linear map `L (D×D)` by gradient descent so the true cluster's
cosine score ranks high, then fold it into the centroids (`c'_k = normalize(L μ_k)`) so inference is
the identical `h · c'` gemv. It is the *steelman* of the falsified whitening result — an exact inner
product against relearned centroids, not an approximation.

## Steps (what the PoC does)

1. Capture `h` + dense argmax over WikiText-2 (cached), map argmax → true cluster.
2. Held-out split. Train `L` (full-rank, and a low-rank `I + UVᵀ` variant) by sampled-softmax
   cross-entropy (true cluster vs random negatives), shrinking toward identity.
3. Fold `L` into the centroids and measure held-out `required_p` vs the cosine baseline.

## Run

```bash
uv run python experimental/learned_metric/learned_metric_poc.py
```

Needs `artifacts/qwen3_0_6b/clusters.npz`. ~2 min (one model pass, cached to `/tmp`; then torch CPU).

## Findings (2026-06-21)

12k WikiText-2 positions, held-out fit/eval split. Lower `required_p` / higher agree = better.

| routing | p50 | p90 | p99 | @128 | @256 | @512 |
|---|---|---|---|---|---|---|
| cosine (shipped) | 4 | 55 | **842** | 95.48% | **97.73%** | **98.57%** |
| learned-L full (D×D) | **2** | 76 | 2644 | 91.92% | 93.75% | 95.55% |
| learned-L low-rank (r=32) | **2** | 96 | 2901 | 91.08% | 93.35% | 95.27% |

**The learned metric trades the tail for the median.** It halves the median rank (4 → 2: it sharpens
the easy, frequent clusters) but **triples p99 (842 → 2644)** and loses ~4pp at every P. Top-1
agreement is decided by the heavy tail (the idiosyncratic tokens `always-score` had to special-case),
so the net is a clear loss. The low-rank variant (65k params) and the identity-shrinkage regularizer
both lose the same way, so this is **not** mere overparameterization — the discriminative objective
*genuinely* reshapes the metric to win the average and lose the tail.

**Why.** A single linear map can only spend its capacity where the training mass is: the frequent
clusters. The tail tokens are rare in calibration data and route badly because of a bad *assignment*
(their embedding sits far from any centroid), which no global routing transform can fix — the same
reason `recall_lift`'s learned-`h` prototype and the MIPS-ranking swaps failed. Sharpening the metric
for the average actively distorts those tail clusters' geometry, pushing them further down.

**Verdict: parked.** This was the one untested routing transform (discriminative vs the falsified
unsupervised ones), and it loses in the now-familiar shape. Conclusion across `recall_lift` +
`whitened_routing` + `factorized_router` + here: **cosine-on-the-embedding-mean is at the ceiling for
any single linear routing pass.** The wins keep routing exact and change the *partition*
(`anisotropic_clustering`, `multiple_assignment`) or sidestep routing entirely (`always-score`). The
only twist left untried is a recall-hinge loss (penalize only when true-cluster-rank > P, so the
metric is never rewarded for the median 4→2 move) — but the failure is tail *generalization*, not the
loss shape, so the prior is that it loses too.
