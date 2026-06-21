# multiple_assignment

**Status: GRADUATED (2026-06-21) — shipped as `turbohead-build-clusters --r 2` + sampling dedup in
`decode_loop._pick`. +1.25pp agree@256 (96.75 -> 98.00) at ~1% end-to-end TPS cost vs the deployed
fp32 head once stage 2 is int8 (`turbohead-splice --head-weight-dtype int8`). The old "~25% decode" was
r=2-on-fp32; int8 halves the gather bytes (8.4 vs 16.8 MB/token) and dissolves it. r=2-int8 vs int8 r=1
is ~1-3% across threads. This folder is the evidence + the int8-vs-fp32 agreement A/B (`build_r2.py`).
See LOG.md 2026-06-21.**

## Objective

Raise top-1 agreement by catching the heavy recall tail: the tokens whose true cluster ranks ~1000th
in the routing order, which no amount of re-scoring can reach.

## The idea (read this if you're new)

Agreement equals stage-1 recall. You hit the dense argmax only when its cluster sits in the top-P
probed clusters. The misses form a heavy tail (p99 required-P around 1165). Re-scoring the clusters
cannot fix that: if you never probe a cluster, no score brings it back.

The classical IVF answer (Jégou et al., PAMI 2011, *multiple assignment*) stops forcing each token
into exactly one cluster. You put it in its top-r nearest clusters. A tail token whose primary cluster
routes badly often sits near a second cluster that routes well. Give it that second home and you catch
it. A position becomes reachable at P when any of the token's r homes lands in the top-P.

## Where the cost lands

Stage 1 (the `[K,D]` centroid gemv plus TopK) does not change: same centroids, same K, same P.
Multiple assignment only grows stage 2, the part that gathers and scores the candidate rows. With r=2
each probed cluster carries `2·cap` members instead of `cap`, so stage 2 gathers twice as many rows.

We first assumed stage 2 was a small slice, which would make r=2 almost free. The profile (below)
proved otherwise: stage 2 is about 28% of a decode step on Qwen3-0.6B, so r=2 costs roughly 25% more
decode time. Read the cost check before you ship this.

## Steps (what the PoC does)

1. Keep the shipped centroids and primary partition.
2. Give every token its assigned cluster plus its (r−1) next-nearest clusters as extra homes.
3. Recompute `required_p` as the min rank over a token's homes. r=1 reproduces the baseline.
4. Report agreement at fixed P, and a cost-matched view (r homes at P/r holds the stage-2 candidate
   budget equal to r=1 at P).

This measures the recall ceiling. A real balanced build is the graduation step.

## Run

```bash
uv run python experimental/multiple_assignment/multiple_assignment_poc.py
```

Needs `artifacts/qwen3_0_6b/{head_W.npy,clusters.npz}`. About 35 s (no re-clustering).

## Findings (2026-06-20)

**Quality.** At fixed P=256, r=2 raises agreement +2.25pp to 99.00% and crushes the tail (p99 1165 to
251, mean 75.7 to 21.3). The cost-matched check (r=2 @128 = 97.17% beats r=1 @256 = 96.75%) shows r=2
still wins even when you pay by halving P. r=2 is the sweet spot; r≥3 spends too much of the probe
budget once you account for the candidate inflation.

**Cost (measured, not assumed).** A single-threaded profile of the shipped fused model puts the
`FlashHeadSelect` op (stage 2) at 6.36 ms/token, 28.5% of the 22.27 ms decode step. r=2 roughly
doubles that gather, adding about 5 to 6 ms, which lands near a 25% decode slowdown. Stage 2 is
memory-bound on the candidate gather (4096 rows of D=1024 pulled from a 622 MB head matrix), so
doubling the row count doubles the bottleneck. Full numbers in [`LOG.md`](LOG.md).

## How it relates to the other levers

It does not stack with `anisotropic_clustering`: both fix the same tail, and r=2 already collapses it,
so the better partition adds almost nothing on top (see `experimental/combinations/`). Pick one. The
decision table lives in `experimental/combinations/README.md`.

## Promotion path

Two regimes, pick by your latency budget:

**Full r=2 @P (the +2.25pp win, costs ~25% decode).** Keep the shipped centroids and `Cnorm`; do NOT
re-run k-means (see the warning below). Give each token its next-best centroid as a 2nd home, growing
`Vmap`/`Wperm` from `cap` to `2·cap` per cluster so K and the stage-1 gemv stay fixed and only the
stage-2 lists grow. Confirm the fused op dedups a token that lands in two probed clusters. Bench
single-threaded against baseline to replace the ~25% estimate with a measured figure.

**Cost-matched r=2 @P/2 (free slice, +0.42pp).** Same 2·cap build, but deploy at P=128 instead of 256.
Stage 2 gathers the same 4096 rows as the shipped P=256/cap=16, and TopK is cheaper at P=128, so it is
a small net speedup. Modest but genuinely free.

**Warning — do not rebuild the centroids.** Tested (`cost_matched_real_build_poc.py`): rebuilding the
centroids over the doubled r=2 memberships BLURS them (mean shift 0.508), and the blur shifts routing
for the whole vocab enough to drop below even the r=1 baseline at matched cost (96.00% vs 96.75%). The
r=2 win only survives on the original SHARP centroids — add soft homes, leave `Cnorm` alone.

A cheaper variant, now tested and parked: give a second home only to the heavy-miss tokens (the set
`turbohead-calibrate-misses` finds), not all of them. It does add far fewer rows than a blanket r=2,
but it loses the bake-off against the already-shipped always-score lever, which targets the same
tokens, is already ~free, and scores higher. See `experimental/targeted_second_home/`.
