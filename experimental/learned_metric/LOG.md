# learned_metric — run log

## 2026-06-21 — NEGATIVE result

Setup: 12k WikiText-2 positions (Qwen3-0.6B `h` + dense argmax, cached), held-out 6k/6k split. Learn a
linear routing map `L (D×D)` by sampled-softmax cross-entropy (true cluster vs 2048 random negatives,
batch 1024, 500 steps, Adam, shrink-to-identity 1e-3, learnable cosine temperature). Fold into the
centroids (`c'_k = normalize(L μ_k)`); eval held-out `required_p`. Tried full-rank and low-rank
`L = I + UVᵀ` (r=32, 65k params).

| routing | p50 | p90 | p99 | @128 | @256 | @512 |
|---|---|---|---|---|---|---|
| cosine (baseline) | 4 | 55 | 842 | 95.48% | 97.73% | 98.57% |
| learned-L full | 2 | 76 | 2644 | 91.92% | 93.75% | 95.55% |
| learned-L r=32 | 2 | 96 | 2901 | 91.08% | 93.35% | 95.27% |

**Finding. The learned metric improves the median (rank 4→2) and wrecks the tail (p99 842→2644), so
top-1 agreement falls ~4pp.** Identical failure signature to the MIPS-ranking POC: optimizing the
average reorders the tail, and top-1 lives in the tail. Low-rank and identity-shrinkage both lose the
same way → not overparameterization; the discriminative objective genuinely makes this trade.

**Why.** A global linear map spends capacity where the training mass is (frequent clusters). The tail
tokens route badly because of a bad *assignment*, which no routing transform can fix; sharpening the
metric for the average distorts those clusters further. Same root cause as `recall_lift`'s
learned-`h` prototype.

**Verdict: parked.** Closes the routing-matrix line — cosine-on-the-embedding-mean is at the ceiling
for any single linear routing pass, learned or fixed. Wins are in the partition or in always-score.
Untried twist (low prior): a recall-hinge loss that ignores the median move; the failure is tail
generalization, not the loss shape.
