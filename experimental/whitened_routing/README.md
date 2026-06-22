# whitened_routing

**Status: PARKED — negative result (whitening hurts routing, badly).**

## Objective

Test whether a learned linear metric on the routing score — folded into the centroids so it's free at
inference — beats plain cosine.

## The idea (read this if you're new)

Stage-1 routing scores clusters by cosine `h · ĉ`. LLM hidden states are **anisotropic**: a few
dimensions carry huge variance and dominate every dot product. The classical nearest-neighbour fix is
a Mahalanobis metric — score with `hᵀ A c` for a learned `A` (the inverse covariance "whitens" the
space, de-emphasising the high-variance dims).

The trick that would make it free: `hᵀ A c = h · (A c)`. Precompute `c' = A c` offline → run time does
the **same** `h · c'` gemv. (A pure *rotation* would be a no-op — rotations preserve inner products —
which is why OPQ-style rotation only helps paired with a product quantizer; see `factorized_router`.)
recall_lift already tried *diagonal* whitening; this is the full-rank generalisation.

## Steps (what the PoC does)

1. Fit mean `μ` and covariance `Σ` on a train split of hidden states.
2. Fold `A = (Σ + λI)⁻¹` into the shipped centroids; sweep the shrinkage `λ`.
3. Measure held-out `required_p`; compare to cosine and to diagonal whitening.

## Run

```bash
uv run python experimental/whitened_routing/whitened_routing_poc.py
```

Needs `artifacts/qwen3_0_6b/clusters.npz`. ~1 min.

## Findings (2026-06-20)

**Whitening loses, badly, and monotonically** — the closer `A` gets to the true `Σ⁻¹` (smaller `λ`),
the worse routing gets (agree@256 falls from 96.65% to ~34%). Diagonal whitening also loses here.

**The useful lesson:** in LLM hidden states the high-variance directions are **signal, not noise** —
they carry what separates clusters. Whitening throws away exactly the information routing needs. The
anisotropy is informative. This *reinforces* `anisotropic_clustering`: put the data-awareness in the
**partition** (respect the parallel/inner-product structure), not in the query metric — and keep the
cosine router. Details in [`LOG.md`](LOG.md).
