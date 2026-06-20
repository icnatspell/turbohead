# combinations

Studies that test whether two or more levers **stack** (gains add up) or **overlap** (they fix the
same thing, so combining them buys little). One script per pairing, named after the levers it tests.
Nothing graduates from here. The output is a decision: which levers to ship together, and which are
redundant.

## Studies

| script | levers tested | verdict |
|---|---|---|
| `aniso_x_multiassign_poc.py` | `anisotropic_clustering` (η) × `multiple_assignment` (r) | do **not** stack — both fix the same tail; ship one |

---

## aniso_x_multiassign — does anisotropic_clustering stack with multiple_assignment?

**Status: answered — they do NOT stack (sub-additive). Ship one, chosen by your latency budget.**

### Why ask

Each lever raises top-1 agreement on its own. They act on different parts: anisotropic_clustering
reshapes the PARTITION, multiple_assignment adds HOMES to whatever partition exists. Different
mechanisms suggest the gains should add. This checks whether they actually do.

### Steps (what the PoC does)

A clean 2×2 with one consistent pipeline (warm-start + 3 balanced passes), varying only `η`
(anisotropy) and `r` (homes per token):

```
(η=1, r=1) control     (η=4, r=1) anisotropic alone
(η=1, r=2) multi alone (η=4, r=2) both
```

Metric: `required_p` = min rank over a token's r homes, on real hidden states. It reuses the actual
`cluster()` (from `anisotropic_clustering`) and `homes_for()` (from `multiple_assignment`), so it
tests the shipping functions rather than a re-derivation.

### Run

```bash
uv run python experimental/combinations/aniso_x_multiassign_poc.py
```

Needs `artifacts/qwen3_0_6b/{head_W.npy,clusters.npz}`. About 3.5 min (builds both partitions).

### Findings (2026-06-20)

**Sub-additive.** At P=256: anisotropic alone +0.52pp, multiple-assignment alone +2.02pp, both
together +2.02pp (not +2.55pp). Anisotropic adds about 0 once r=2 is in. Both target the same heavy
tail, and r=2 already collapses it (mean req-P 70.5 to 20.0, p99 1042 to 226), so the better partition
has nothing left to rescue. Full table in [`LOG.md`](LOG.md).

### Decision table (which lever, when)

A separate cost check (in `multiple_assignment/LOG.md`) measured the price of r=2: stage 2 is about
28% of a decode step, so r=2 costs roughly 25% decode speed. Anisotropic clustering costs nothing. The
choice is a quality/latency trade.

| Your situation | Pick | Result | Cost |
|---|---|---|---|
| Default. You want a free quality bump and a byte-identical graph shape. | `anisotropic_clustering` (η=4) | agree@256 96.7% → 97.6% | none (same gemv, same candidate count) |
| You need the highest agreement and can spend ~25% decode latency. | `multiple_assignment` (r=2) | agree@256 → 99.0%, tail crushed | ~25% slower decode single-threaded; confirm you still beat your dense-head baseline |
| You are tempted to do both. | Don't. | both = r=2 alone at P=256 | you pay r=2's cost and gain nothing extra |

Middle ground (tested, parked): give a second home only to the heavy-miss tokens, not every token. It
does add far fewer rows, but it loses to the already-shipped always-score lever (same tokens, ~free,
higher agreement). See `experimental/targeted_second_home/`.
