# anisotropic_clustering

**Status: WIN, free — the default agreement lever (+0.85pp @256, tail collapses, zero inference
cost). Preferred over `multiple_assignment` unless you can spend ~25% decode latency for its larger
gain; see `experimental/combinations/README.md` for the decision table.**

## Objective

Lift top-1 agreement (= stage-1 recall) **without changing inference cost**, by building a better
partition of the vocabulary into clusters.

## The idea (read this if you're new)

FlashHead returns the dense model's argmax token exactly when that token's cluster is among the top-P
clusters stage 1 probes. So agreement is decided by the partition and the routing score.

`build_clusters.py` builds the partition with plain k-means: it minimises Euclidean error
`‖e − c‖²` between a token embedding `e` and its centroid `c`. But we don't *search* by Euclidean
distance — we search by inner product `⟨h, e⟩` (hidden state · embedding). For an inner product, the
quantization error that actually moves the score is the component **parallel** to `e`; the orthogonal
part barely matters. Plain k-means weights both equally, so it spends centroid accuracy on the
direction that doesn't affect ranking.

ScaNN (Guo et al., ICML 2020, *Anisotropic Vector Quantization*) fixes this by penalising the
parallel error more. Assign token `x` to the centroid minimising

```
L(x, c) = ‖x − c‖²  +  (η − 1) · ((x − c) · x̂)²        x̂ = x / ‖x‖
```

`η = 1` is plain k-means (sanity check); `η > 1` reshapes clusters to preserve inner-product
*ranking* — exactly what recall needs. **Inference cost is unchanged**: same `(D,K)` centroid matrix,
same stage-1 gemv. Only the offline clustering changes.

## Steps (what the PoC does)

1. Warm-start from the shipped centroids.
2. Run a few balanced refinement passes using the anisotropic assignment score, sweeping `η`.
3. Recompute each centroid as the member mean (a deliberate simplification — see below).
4. Measure `required_p` (rank of each position's true cluster) on real hidden states; report
   agreement at P = 128/256/512.

**Centroid update:** the PoC builds both the member-mean centroid and ScaNN's closed-form
parallel-weighted centroid (`centroids_scann`), and the mean wins at the deploy P (see Findings). We
deploy COSINE routing, which keeps only the centroid's direction; the closed form tunes magnitude and
the parallel component, and normalising the centroid discards exactly that. So the member mean is the
operating-point optimum here, not a lower bound. Ship the mean.

## Run

```bash
uv run python experimental/anisotropic_clustering/anisotropic_clustering_poc.py 2 4 8   # η values
```

Needs `artifacts/qwen3_0_6b/{head_W.npy,clusters.npz}`. ~2.3 min per η.

## Findings (2026-06-20)

`η = 4` beats the controlled baseline (same iteration budget) on every metric. The win is in the
**tail**: mean required-P 70.5 → 46.4 (−34%), p99 1042 → 645 (−38%); agreement at the deploy P=256
+0.52pp vs the matched control (+0.85pp vs shipped). Best range `η ∈ [4, 8]`. Full numbers and
graduation steps in [`LOG.md`](LOG.md).

**Closed-form centroid: tested, lost.** ScaNN's parallel-weighted centroid (`centroids_scann`) is
equal-or-worse than the member mean at the deploy P (97.38% vs 97.60% @256, 93.65% vs 95.17% @128),
because cosine routing keeps only the direction and normalisation discards the magnitude the closed
form tunes. Ship the member mean.

**Promotion path:** add `--eta` to `surgery/build_clusters.py` (default 1.0 = current behaviour),
keep the member-mean centroid (the closed form was tested and lost — see Findings), rebuild a full
artifact, confirm the end-to-end agreement gain on the spliced model.

**Does not stack with `multiple_assignment`:** both fix the same tail, so combining them adds almost
nothing over `multiple_assignment` alone (see `experimental/combinations/`). They are alternatives, picked
by your latency budget.
