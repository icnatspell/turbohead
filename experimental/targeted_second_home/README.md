# targeted_second_home

**Status: PARKED — works, but dominated by the shipped always-score lever for this workload.**

## Objective

Rescue the heavy miss tail (the frequent tokens FlashHead routes badly) more cheaply than the two
existing options: blanket `multiple_assignment` (r=2, which doubles every cluster and costs ~25%
decode) and the shipped always-score lever (a flat list scored every step). The idea: give a second
home cluster only to the tokens that actually miss, so their extra row is gathered only when that
second cluster is probed.

## The idea (read this if you're new)

Top-1 agreement == stage-1 recall: FlashHead returns the dense argmax only when that token's cluster
lands in the top-P probed clusters. About half the misses are the same frequent tokens every time
(function words, punctuation) sitting far from their cluster's centroid, so routing never ranks them
in. Two ways to rescue exactly those tokens:

- **always-score (shipped, `recall_lift` lever 4):** score the token's weight row on *every* step,
  regardless of routing (rides the graph's Wspec/EOS path). Flat cost: N extra rows per step.
  Unconditional rescue.
- **targeted second home (this experiment):** add the token to its next-best cluster, so it is
  reachable through a *second* route. Amortized cost: its row is gathered only on steps where that
  second cluster is in the top-P. Conditional rescue.

The honest framing: this does **not** compete with a blanket r=2. It competes with always-score, which
already targets the same tokens and already ships. So the PoC is a head-to-head between the two.

## Steps (what the PoC does)

1. Collect hidden states + dense argmax on ~12k WikiText-2 positions; split 50/50 fit / eval.
2. Fit the most-missed token set on the train half (same calibration the shipped lever uses).
3. For N in {64, 256, 1024, 4096}, rescue that set two ways and measure on the held-out eval at P=256:
   rescue fraction of the misses, resulting agreement, and extra rows/step.
   - always-score: caught iff the token is in the set (flat N rows/step).
   - second home: caught iff the token's second-nearest cluster is in that step's top-P (amortized).

## Run

```bash
uv run python experimental/targeted_second_home/targeted_second_home_poc.py
```

Needs `artifacts/qwen3_0_6b/{head_W.npy,clusters.npz}`. ~1–2 min (one model pass).

## Findings (2026-06-20)

Baseline agree@256 = 97.47%. Both mechanisms plateau at N=64.

| N=64 | rescue | agree@256 | rows/step |
|---|---|---|---|
| always-score | 52.0% | **98.78%** | 64 |
| targeted second home | 43.4% | 98.57% | **7.4** |

Second home captures ~84% of always-score's lift at ~1/9 the extra rows, so per row it is ~7× more
efficient. It loses anyway, because:

1. The cost it saves doesn't exist — always-score's 64 rows/step sit on top of the 4096 candidate rows
   already gathered (1.6% more), through the existing Wspec path with no graph change.
2. Its rescue is conditional (only when the second cluster is probed), so it stays 0.22pp below
   always-score at every N and can't reach its ceiling.
3. Its one real edge — cheap rescue at large N — never applies: the lift plateaus at N=64 (the rest of
   the tail is idiosyncratic one-offs), so a large set is never wanted.
4. It costs build complexity: variable cluster size breaks the equal-`cap` kernel, plus dedup when a
   token lands in two probed clusters. Always-score needs none of that.

**Verdict: parked.** Always-score already ships, is already ~free, and gives strictly higher
agreement. Full table and the "when it could come back" note in [`LOG.md`](LOG.md).
