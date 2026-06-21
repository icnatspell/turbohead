# multiple_assignment — run log

## 2026-06-20 — recall ceiling sweep

Setup: shipped centroids + primary partition unchanged. Each token gets its assigned cluster plus
its (r-1) next-nearest clusters as extra homes; required-P recomputed as the MIN rank over homes.
r=1 reproduces the baseline exactly. Eval on 4000 real Qwen3-0.6B hidden states. This is the recall
CEILING (assumes a clean top-r home set); a real balanced build is the graduation step.

| r homes | p50 | p90 | p99  | mean | agree@128 | @256   | @512   |
|---------|-----|-----|------|------|-----------|--------|--------|
| 1 (base)| 5   | 81  | 1165 | 75.7 | 93.67%    | 96.75% | 97.85% |
| **2**   | 2   | 46  | 251  | 21.3 | 97.17%    | 99.00% | 99.58% |
| 3       | 2   | 41  | 210  | 17.8 | 97.72%    | 99.17% | 99.72% |
| 4       | 1   | 34  | 174  | 15.0 | 98.28%    | 99.50% | 99.80% |

cost-matched to r=1 @256 (equal stage-2 candidate budget P*cap*r):
- r=1 @256 -> 96.75% (reference)
- r=2 @128 -> 97.17%   (wins)
- r=3 @85  -> 95.53%   (loses — too few clusters probed)
- r=4 @64  -> 94.85%   (loses)

**Finding.** Biggest agreement lever found so far. At FIXED P=256, r=2 lifts agree +2.25pp to 99.00%
and crushes the tail (p99 1165 -> 251, mean 75.7 -> 21.3). The reason re-scoring experiments couldn't
touch this: the misses are tokens whose true cluster is never probed — a second home fixes exactly
that. r=2 is the sweet spot: in the cost-matched view (the honest "pay for it by halving P") only r=2
still beats baseline; r>=3 spends too much of the probe budget.

**Cost.** stage 1 (the dominant [K,D] gemv + TopK) is UNCHANGED — same centroids, K, P. Only stage 2
(Gather + dot over the candidate rows) grows ~r x: r=2 doubles the candidate count from P*cap to
~2*P*cap. Whether r=2 @256 is "near free" in wall-clock depends on stage-2's share of the step;
needs an end-to-end timing check (see next steps). Stage 2 is the cheap stage, so it should be small.

### Next steps if it graduates
- **Real balanced build.** `build_clusters.py` must place each token in r clusters while keeping
  per-cluster size ~cap (so K grows, or cap grows to r*cap). Growing cap keeps K and the gemv fixed
  (stage 1 stays put) at the price of larger stage-2 lists.
- **Dedup in stage 2.** A token in 2 probed clusters is gathered twice; the onnx backend's scatter
  into the -1e9 base already collapses duplicates. Confirm the fused op's shortlist dedups too, or
  accept the harmless duplicate score.
- **Cheaper variant.** Give a second home only to the heavy-miss tokens (the set
  `turbohead-calibrate-misses` already finds), not all tokens. Likely captures most of the 99% with
  far fewer extra candidate rows than a blanket r=2.

## 2026-06-20 — cost check: stage 2 is NOT a small slice (r=2 is not near-free)

Profiled the shipped fused model single-threaded (`turbohead-decode artifacts/qwen3_0_6b/fused
--profile --max-new 96`):

| op (per token)                  | ms/step | share of step |
|---------------------------------|---------|---------------|
| decode step (model_run median)  | 22.27   | 100%          |
| FlashHeadSelect (stage 2)       | 6.36    | 28.5%         |
| TopK (stage 1)                  | 0.09    | 0.4%          |

Stage 2 costs 6.36 ms because it gathers P*cap = 4096 candidate rows of D=1024 from the 622 MB head
matrix. That gather is memory-bound and scattered, so it dominates the head even though the dot
product is trivial. r=2 doubles the candidate count to 8192 rows, so it roughly doubles this gather:
about +5 to +6 ms, pushing the step from 22.3 ms to ~28 ms, near a 25% single-threaded slowdown. This
is an upper bound, since FlashHeadSelect's fixed work (topk read, output scatter) does not double.

**Correction to the ceiling log above.** The "+2.25pp at 2x the cheap stage 2" framing assumed stage
2 was small. On Qwen3-0.6B it is ~28% of the step, so r=2 trades roughly 25% decode speed for the
agreement gain. Take it when quality outranks latency and you still beat your dense-head baseline.
`anisotropic_clustering` (eta=4) gives a smaller agreement gain at zero speed cost, so it is the
better default; see `experimental/combinations/README.md` for the decision table.

## 2026-06-20 — cost-matched r=2 with a REAL rebuilt partition (does the win survive a real build?)

The ceiling sweep above measured r=2 @128 = 97.17% (> r=1 @256 = 96.75%, same 4096 candidate rows)
but cheated on the build: it kept the SHIPPED r=1 centroids and tacked on next-nearest homes. A real
r=2 deploy rebuilds the centroids over the doubled memberships, which blurs them — every centroid
becomes a mean over ~2*cap members, shifting routing for ALL tokens, not just the tail. This run
rebuilds the centroids so the (-) of the blur shows up alongside the (+) of the second home.
(`cost_matched_real_build_poc.py`, 4000 real hidden states, all three gather 4096 rows.)

| variant | centroids | P | candidate rows | agree |
|---|---|---|---|---|
| A — r=1 (deployed reference) | shipped | 256 | 4096 | 96.75% |
| B — r=2 soft homes | shipped (sharp) | 128 | 4096 | **97.17%** |
| C — r=2 real build | rebuilt (blurred, mean shift 0.508) | 128 | 4096 | 96.00% |

**Finding: rebuilding the centroids LOSES.** C is -1.18pp under B and -0.75pp under even the r=1
baseline. The blur (mean centroid shift 0.508 in embedding space) costs more recall across the whole
vocab than the second home buys back on the tail. So the cost-matched r=2 win is real but fragile: it
only exists if you keep the shipped SHARP centroids and add soft homes on top.

**Promotion path simplifies (no re-clustering).** Don't run a balanced 2-assignment k-means. Keep the
shipped centroids and `Cnorm` exactly as built. Give each token its next-best centroid as a 2nd home
(grow `Vmap`/`Wperm` from cap to 2*cap per cluster; stage-1 gemv and K untouched). Deploy at P=128:
stage 2 gathers the same 4096 rows as the shipped P=256/cap=16, and TopK is cheaper at P=128, so it is
a small NET speedup. Payoff is modest (+0.42pp agree, 96.75% -> 97.17%) but genuinely free. The big
+2.25pp@256 r=2 still costs the ~25% from the cost check above; the cost-matched variant is the free slice.

## 2026-06-21 — REAL BALANCED r=2 BUILD + int8 dissolves the cost (the graduation run)

`build_r2.py`. Built the actual fixed-width r=2 table the fused op needs: kept the SHARP shipped
centroids, ran a *balanced* 2nd assignment (greedy-capacity Lloyd, each token's primary cluster
excluded) so every cluster gets exactly cap seconds -> table grows `(K,cap,D)` -> `(K,2cap,D)`,
cap 16 -> 32. The op reads cap from `Wperm.Shape()[1]`, so no C++ change. Spliced `fused_r2`
(fp32) and `fused_r2_q8` (int8); both load, run, decode coherent text.

Agreement, 4000 real Qwen3-0.6B positions:

| P   | r=1 fp32 (shipped) | r=2 fp32 | r=2 int8 | r2 gain | int8 drop |
|-----|--------------------|----------|----------|---------|-----------|
| 128 | 93.67%             | 96.97%   | 96.28%   | +3.30pp | -0.70pp   |
| 256 | 96.75%             | 98.78%   | 98.00%   | +2.02pp | -0.78pp   |

Two questions answered:
1. **Does balancing the seconds keep the ceiling?** Yes. fp32 r=2 @256 = 98.78%, only 0.22pp under the
   unbalanced ceiling (99.00%). Constraining each cluster to exactly cap seconds barely hurts — a
   token's forced 2nd home is usually still a good route. Balancing is nearly free.
2. **Does int8 survive?** Yes, with a small give-back. int8 costs 0.78pp (near-tied candidate
   argmaxes flip under per-row quant noise), but r=2-int8 still ships **+1.25pp over shipped fp32 r=1**.

End-to-end speed, single thread (this run's box was lightly loaded, so read ratios not absolutes):

| head                 | tok/s @1t | vs shipped |
|----------------------|-----------|------------|
| fused fp32 r=1 (shipped) | 60.7  | 1.00x      |
| fused_r2 fp32 r=2    | 58.6      | 0.96x      |
| fused_q8 int8 r=1    | 63.2      | 1.04x      |
| **fused_r2_q8 int8 r=2** | **61.5** | **1.01x** |

**The "+2.25pp costs ~25% decode" verdict is DISSOLVED.** That was r=2-on-fp32. int8 halves the
stage-2 bytes (8.4 vs 16.8 MB/token), so int8 r=2 is speed-NEUTRAL vs shipped fp32 r=1 (1.01x) while
adding +1.25pp. fp32 r=2 alone costs only ~4% here (the doubled gather is bandwidth-hidden at 1
thread), not 25% — the old estimate over-counted by assuming the gather doubles the whole 6.36 ms op
including its fixed work. On the ARM board the head holds a bigger serial share, so int8 r=2 should
pull AHEAD of shipped, not just break even. Earlier P=512 proxy (`fused_q8_P512`, same 8192 rows)
measured 1.06x, bracketing this 1.01x as "roughly neutral, leaning faster".

**Remaining before this fully ships:** dedup. A token in two probed clusters is scored twice in the
shortlist. Harmless for greedy (same id/logit wins argmax — verified, decode coherent). For sampling
it double-weights that token; needs a dedup in the fused op (C++) or the Python shortlist softmax. Not
done here. Graduation into core also needs the balanced-2nd-assignment wired into `build_clusters.py`
behind an `--r 2` flag.
