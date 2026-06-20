# factorized_router

**Status: PARKED — negative result (PQ routing too lossy to convert into an agreement win).**

## Objective

Stage 1 (the `[K,D]` gemv) is the *dominant* head cost and gates how large P can be. Test whether a
**cheaper** router — product-quantized centroids — preserves the cluster ranking well enough to
**raise P for free** and thereby lift recall.

## The idea (read this if you're new)

Agreement is recall-limited, and the cheapest way to raise recall is to probe more clusters (raise P).
But P is gated by the stage-1 budget. Classical tool: **Product Quantization + ADC** (Jégou et al.,
PAMI 2011; the inverted multi-index, Babenko & Lempitsky 2012, is the routing form). Split each
centroid into m sub-vectors, quantize each subspace into a tiny codebook, and approximate
`h · c_k` as a sum of m table lookups. Cost drops from `K·D` to `~Ksub·D + K·m` (tens of × fewer
multiply-adds). If the ranking survives, you spend the savings on a bigger P.

## Steps (what the PoC does)

1. Product-quantize the `(K,D)` centroid matrix (m subspaces, 256 codewords each).
2. Route by ADC table lookups; sweep m.
3. Report recall of the true cluster vs exact cosine, **and** the stage-1 FLOP ratio.
4. Decisive test: does PQ at a bigger P (afforded by the FLOP savings) beat exact at the deploy P?

## Run

```bash
uv run python experimental/factorized_router/factorized_router_poc.py
```

Needs `artifacts/qwen3_0_6b/clusters.npz`. ~45 s.

## Findings (2026-06-20)

**PQ routing is cheap (23–32× fewer FLOPs) but far too lossy.** PQ m=16 @256 = 70.6% vs exact 96.75%
(−26pp). The decisive test fails: PQ m=16 **@1024 = 86.52%** — still ~10pp **below** exact **@256**,
despite probing 4× more clusters. The approximation reorders clusters at the top-P boundary and
destroys the hard tail (same failure as `hierarchical_stage1`). The shipped int4 `MatMulNBits` stage 1
is already fast *and* holds 100% argmax, so there's no opening. Details in [`LOG.md`](LOG.md).

**Meta-finding (shared with `whitened_routing`):** approximating the routing inner product loses every
time — recall is razor-sensitive to the exact `h·c` order near the top-P cut. The levers that win
(`anisotropic_clustering`, `multiple_assignment`) keep routing **exact** and change the **partition**.
