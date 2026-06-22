# combinations: anisotropic_clustering x multiple_assignment — run log

## 2026-06-20 — do #1 and #2 stack? NO (sub-additive)

Clean 2x2, one pipeline (warm-start + 3 balanced passes), only eta and r vary. required_p = min rank
over the token's r homes, on 4000 real Qwen3-0.6B hidden states.

| variant              | p50 | p90 | p99  | mean | @128   | @256   | @512   |
|----------------------|-----|-----|------|------|--------|--------|--------|
| eta=1 r=1 (control)  | 4   | 72  | 1042 | 70.5 | 94.62% | 97.08% | 98.08% |
| eta=4 r=1 (aniso)    | 3   | 68  | 645  | 46.4 | 95.17% | 97.60% | 98.67% |
| eta=1 r=2 (multi)    | 2   | 40  | 226  | 20.0 | 97.52% | 99.10% | 99.55% |
| eta=4 r=2 (stacked)  | 2   | 42  | 234  | 19.2 | 97.65% | 99.10% | 99.60% |

agree@256 deltas vs control: aniso +0.52pp, multi +2.02pp, **joint +2.02pp** (sum of solos +2.55pp).

**Finding. Sub-additive: anisotropic adds ~0 on top of r=2 at P=256.** Stacked @256 (99.10%) equals
multiple-assignment alone (99.10%). Anisotropic contributes only a hair at smaller P (@128
97.52 -> 97.65) and on the mean (20.0 -> 19.2).

**Why.** Both levers target the SAME heavy tail. Multiple-assignment (r=2) already collapses it
(mean 70.5 -> 20.0, p99 1042 -> 226), so there is almost nothing left for the better partition to
rescue. They are redundant, not complementary.

**Takeaway.** Ship ONE; they are redundant. The pick is a quality/latency trade once you fold in the
measured cost of r=2 (stage 2 is ~28% of the step, so r=2 costs ~25% decode speed; see
`multiple_assignment/LOG.md`): `anisotropic_clustering` (eta=4) is the free default at +0.85pp;
`multiple_assignment` (r=2) buys 99.0% but pays ~25% decode speed. Decision table in the README.
