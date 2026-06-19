# How our clustering differs from the FlashHead paper

The FlashHead paper (Tranheden et al., Embedl, arXiv 2603.14591) reports that clustering one
model took 4 hours on an A40 GPU. Our `build_clusters.py` finishes in minutes on a CPU. A GPU
runs each iteration faster than a CPU, so the gap is not hardware. We do less work, and we
make different algorithm choices. This doc records what differs, why we are still fast, what
it costs in quality, and which knobs raise top-1 agreement.

Written for a junior engineer. Read `turbohead/surgery/build_clusters.py` alongside it.

## The headline: iteration budget

Each k-means iteration is dominated by one matmul, `X @ C.T`, that scores every token
against every centroid. For Qwen3-0.6B that is `151936 tokens × 9496 clusters × 1024 dims`,
about 1.5 TFLOP. A BLAS-threaded CPU runs that in roughly 15 seconds.

| | iterations | est. time |
|---|---|---|
| paper | "within 1000" | ~1000 × 15 s ≈ 4 hours |
| ours  | 15 settle + 5 balanced ≈ 20 | ~20 × 15 s ≈ 5 minutes |

The arithmetic reproduces their 4 hours from the iteration count alone. We run about 20
iterations (`kmeans()` uses `settle_iters=15`, `balanced_iters=5`). That single factor
explains most of the speedup.

### Do those 20 iterations converge?

Almost, then they crawl. The settle shift (how far centroids move per pass) from a real
build (Qwen3-1.7B):

```
iter 0: 55.9   iter 5: 4.65   iter 9: 1.97   iter 12: 1.48   iter 14: 1.12
```

The big gains land by iter 9. After that the centroids still drift a little (1.1, not 0), so
we stop before full convergence. The cost of stopping early shows up as clustering quality,
covered below.

## The three differences

### 1. Iteration budget (biggest)

Covered above: ~20 passes vs up to 1000.

### 2. Vectorized balanced assignment, run ~6 times with an early bail

Both methods enforce equal cluster sizes (every cluster holds exactly `cap = V/K` tokens),
because a dense, equal-size cluster-to-token map lets the inference kernel gather rows with
plain arithmetic instead of ragged, masked lookups.

How we enforce it (`balanced_assign()`):
- Score all tokens against all clusters in chunked BLAS matmuls.
- Sort every token's best bid by score, then accept the top free-slots bidders per cluster.
  This is a one-shot auction, vectorized except for one O(V) Python pass per round.
- Bail out of the rounds once they stall (`acc_tokens.size <= 4*cap`).
- Place the remainder with a fast block matmul tail (`X[blk] @ C.T`, mask full clusters).

We call this about 6 times total (once, then once per balanced iteration).

The paper describes the constraint differently: when a cluster overflows, "its
lowest-similarity members are reassigned greedily to clusters that still have available
slots." That reads as a serial overflow redistribution run inside every iteration. Serial
reassignment over a 128k vocabulary, up to 1000 times, is the kind of step that costs hours.

### 3. Euclidean k-means, not spherical

We minimize squared Euclidean distance:

```
argmin ||x - c||^2  ==  argmax (x·c - 0.5||c||^2)   # build_clusters.py:19
```

and normalize only the final centroids, once, for routing (`build()`, line 123). The paper
runs spherical k-means: it measures similarity with cosine and re-normalizes centroids to the
unit sphere every iteration. Their stated reason is that token-embedding semantics live in the
*direction* of the vector, not its length.

This is mostly bookkeeping cost, not the main speed difference. It matters more for quality
(next section).

## What the shortcuts cost in quality

So far, nothing measurable. Our top-1 agreement (flash top-1 matches the dense head's argmax)
lands at 94-98% across P=128-512, which matches the paper's top-k containment. The early stop
and the Euclidean objective do not visibly hurt at current settings.

The risk is latent. Push to larger `K`, a harder or multilingual vocabulary, or a model whose
balanced rounds stall early, and agreement can drop. The levers below are where to look.

## Levers to raise top-1 agreement (ranked)

1. **Switch to spherical (cosine) clustering. Untested, most principled.**
   We cluster by Euclidean distance but route by cosine and refine by raw dot product. That is
   a metric mismatch. Large-norm embeddings can pull Euclidean centroids around, so groups
   form partly by magnitude instead of direction. Aligning the clustering metric with the
   routing metric (normalize embeddings before clustering, normalize centroids each iteration)
   is a more likely win than more iterations. We have not measured it. Test it first.

2. **Check the early-stall tail fill. A crude-placement risk.**
   When the balanced rounds stall, `balanced_assign` bails and places the rest with a greedy,
   per-token, order-dependent block fill. For a model that stalls early, a large share of the
   vocabulary gets that cruder placement. Measure how much vocab flows through the tail (count
   tokens placed after the `sequential finish` log line) before trusting agreement on a new
   model.

3. **Raise `settle_iters`. Cheap, diminishing.**
   The shift curve shows big gains by iter 9 and slow drift after. Bumping 15 to 30-40 costs
   about 15 s per added pass and buys a small improvement. Worth trying, not a fix.

4. **Raise `balanced_iters`. Untested, likely marginal.**
   Each balanced pass re-derives centroids from the balanced members and re-assigns. More
   passes may tighten the constrained solution.

Note the one place we already match the paper's quality recipe: exact equal-size clusters.
The paper's ablation (their Table 6) shows equal clusters improve both accuracy and latency
over unequal ones, and we enforce exact balance (`assert (free == 0).all()`).

## Caveat on the comparison

The "4 hours" is the paper's single number for one model. It may bundle more than one run, or
converge the spherical reassignment to a tighter tolerance than our fixed 20 passes. The
order-of-magnitude match on the iteration arithmetic is solid. The exact ratio is not the
point; the iteration budget and the assignment method are.
