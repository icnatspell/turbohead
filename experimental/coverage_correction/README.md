# coverage_correction

**Status: PARKED — nothing to promote. The honest report it argued for already ships in core; the
one novel piece (corrected-Z denominator) moves PPL by 0.01.**

## Objective

FlashHead scores only the `P·cap` probed candidates, so a true token in an unprobed cluster gets ~0
probability. That inflates the naive full-distribution PPL and makes it unreadable as an absolute
number. This experiment asked: can we make PPL honest cheaply, using each cluster's mean-embedding
centroid `μ_k` (so `μ_k·h` is the cluster's mean logit) to (a) add the unprobed clusters' mass to the
softmax denominator `Z`, and (b) hand an unprobed true token a non-zero probability?

## What the PoC measures

On 1000 real Qwen3-0.6B positions at P=256, NLL/PPL vs the exact full-V softmax (gold), split into
probed vs unprobed targets:

| method | PPL | probed-only | unprobed-only |
|---|---|---|---|
| gold (dense) | 11.12 | 5.92 | 1639 |
| trunc (current FlashHead) | 10276 | 5.42 | ~1e30 |
| corr_full (deployable: mean-logit numerator) | 33.10 | 5.43 | 5.6e7 |
| corr_cal (corr_full + one offline offset δ) | 9.76 | 5.43 | 1026 |
| corr_Zonly (ceiling: cheat exact logit) | 9.76 | 5.43 | 1026 |

## Findings (2026-06-20)

**The honest report needs no correction, and it already exists.** Splitting probed vs unprobed shows
the truncated denominator is already fine on the probed 88.8% (5.42 vs the Z-corrected 5.43 — a 0.01
PPL difference; the mean-logit tail is a lower bound on the true tail mass by Jensen, so it barely
moves `Z`). The whole inflation is the unprobed 11.2% getting ~0 probability. So the right thing to
report is **covered PPL + coverage%**, two numbers — and `turbohead-ppl` and `turbohead-head-quality`
**already print exactly that** (`flash_ppl` returns `(full, covered, coverage)`).

**The δ calibration is a metric fudge, skip it.** `corr_cal` matching the `corr_Zonly` ceiling is
near-tautological: `δ = mean(L[y] − μ_cl)` over unprobed positions, added back, so the aggregate
mean-NLL matches gold *by construction*. It hides the coverage hole (11.2% of true tokens are
unrepresentable) behind a fitted constant, making FlashHead look near-lossless on likelihood when it
isn't. A reader is better served by "covered PPL 5.4, coverage 88.8%" than by a single δ-tuned 9.76.

**Verdict: nothing to promote.** The covered-PPL + coverage% report (the honest framing) is already in
core. The corrected-Z denominator is a 0.01 PPL no-op. Building it would change no decision and no
published number. `docs/RESULTS.md`'s "read PPL within-model only" guidance stands — coverage caps the
absolute number, and no cheap denominator trick changes that.

## Run

```bash
uv run python experimental/coverage_correction/coverage_correction_poc.py
```

Needs `artifacts/qwen3_0_6b/{head_W.npy,clusters.npz}`. ~1 min (one model pass).
