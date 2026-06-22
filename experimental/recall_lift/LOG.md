# recall_lift — run log

Lever 4 (always-score) shipped; levers 1–3 parked. See `docs/IDEAS.md` §3,5,8 for the original POC.

## 2026-06-21 — frontier: does always-score buy a TPS-saving P cut? (`frontier_poc.py`)

Question (from the routing-matrix dead-ends): all recall wins now keep routing exact; always-score is
the cheap one. Can we run a LOWER P and lean on always-score to recover the recall, for a TPS win?

**Recall side — yes, and the lift grows as P shrinks.** 12k WikiText-2 positions, held-out, list
calibrated on train at P=256. agree@P, always-score list size N:

| P | N=0 | N=64 | N=128 | lift(N=64) |
|---|---|---|---|---|
| 32 | 84.15% | 89.40% | 89.55% | +5.25% |
| 64 | 91.32% | 94.78% | 94.90% | +3.47% |
| 128 | 95.48% | 97.60% | 97.70% | +2.12% |
| 192 | 96.87% | 98.28% | 98.37% | +1.42% |
| 256 | 97.73% | 98.73% | 98.78% | +1.00% |
| 512 | 98.57% | 99.20% | 99.23% | +0.63% |

The lift is NOT constant — it grows from +0.6pp @P=512 to +5.3pp @P=32 (more of the shrinking-P miss
pool is the fixed chronic-misrouter set). N=128 ≈ N=64 everywhere (the plateau holds). **With
always-score N=64, P=192 already matches the P=256 no-list baseline** (98.28% vs 97.73%), a 25% cut in
stage-2 rows; P=128 nearly matches it (97.60%).

**TPS side — the P cut is nearly free on onnx but worthless on fused (the deployed backend).** Real
decode, Qwen3-0.6B, threads=1, median of 7, re-spliced at each P:

| P | onnx tok/s | fused tok/s |
|---|---|---|
| 256 | 48.9 | 54.2 |
| 192 | 50.9 (+4.1%) | 55.3 (~+2%) |
| 128 | 53.7 (+9.8%) | 55.2 (~+2%) |

On **fused** (shortlist-out, the default), lowering P moves TPS ~2% — within noise. The fused op
already skips the O(V) scatter and the P·cap gather is cheap, so the head is dominated by the
P-INDEPENDENT stage-1 int4 gemv. P is simply not the bottleneck there. On **onnx** (logits-out) the
P·cap stage-2 matmul is a real share, so the cut buys ~4–10%, but onnx is the slower backend anyway.

**Conclusion.** always-score shifts the whole recall-vs-P frontier up, most at low P — but on the fast
path that does NOT convert to TPS, because P is already cheap on fused. So use always-score to **raise
recall for free at the current P** (it's not even enabled in the current build — no `always_score.npy`
in the artifact), not to chase a P-cut speedup the fused backend won't pay out. This reconfirms the
stage-1-dominant head picture: the lever for fused TPS is the stage-1 gemv / body, not P.
