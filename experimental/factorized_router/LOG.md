# factorized_router — run log

## 2026-06-20 — NEGATIVE result

Setup: product-quantize the (K,D) centroid matrix into m subspaces, ksub=256 codewords each; route by
ADC table lookups instead of the exact gemv. Measure recall of the true cluster vs exact cosine, plus
the stage-1 FLOP ratio. Question: do the FLOP savings buy a big enough P to beat exact?

| router          | @128   | @256   | @512   | @1024  | stage-1 FLOPs |
|-----------------|--------|--------|--------|--------|---------------|
| exact (cosine)  | 93.67% | 96.75% | 97.85% | 98.88% | 1.0x          |
| PQ m=4          | 55.25% | 62.58% | 70.70% | 78.00% | ~32x fewer    |
| PQ m=8          | 57.40% | 64.90% | 72.90% | 80.53% | ~29x fewer    |
| PQ m=16         | 62.50% | 70.60% | 79.15% | 86.52% | ~23x fewer    |

**Finding. PQ routing is cheap but far too lossy.** PQ m=16 @256 = 70.6% vs exact 96.75% (-26pp).
The decisive test is whether spending the FLOP savings on a bigger P beats exact: it does not.
PQ m=16 @1024 = 86.52% is still ~10pp BELOW exact @256, despite probing 4x more clusters. The
approximation reorders clusters near the top-P boundary and destroys the hard tail -- the same
failure hierarchical_stage1 hit ("coarse prune drops the hard tail").

**Also.** The shipped stage-1 already runs as int4 MatMulNBits, which holds 100% argmax (lossless
ranking) and is fast. PQ-ADC is both lossy AND cache-bound table lookups whose ORT speed vs int4 is
unproven. No path to a net win.

**Verdict: parked.** Routing recall is razor-sensitive to the exact h.c order near the top-P cut;
approximating that inner product (here PQ, in whitened_routing a learned metric) loses every time.
The levers that win (anisotropic_clustering, multiple_assignment) keep routing EXACT and change the
PARTITION instead. That is the meta-finding from this batch.
