# anisotropic_clustering — run log

## 2026-06-20 — first sweep

Setup: warm-start from the shipped centroids, 3 balanced refinement passes, member-mean centroids,
eval `required_p` on 4000 real Qwen3-0.6B hidden states (WikiText-2). Member-mean centroid keeps the
test conservative (ScaNN's closed-form parallel-weighted centroid would only help more).

`eta=1.0` is the **control**, not the shipped baseline: it holds the iteration budget + fp32 fixed,
so the gap between it and `eta>1` is purely the anisotropy effect. (Shipped baseline sits a touch
below the control only because the control runs 3 extra refinement passes in fp32.)

| variant            | p50 | p90 | p99  | mean | agree@128 | @256   | @512   |
|--------------------|-----|-----|------|------|-----------|--------|--------|
| shipped baseline   | 5   | 81  | 1165 | 75.7 | 93.67%    | 96.75% | 97.85% |
| eta=1.0 (control)  | 4   | 72  | 1042 | 70.5 | 94.62%    | 97.08% | 98.08% |
| eta=2.0            | 5   | 86  | 786  | 48.6 | 93.67%    | 96.85% | 98.40% |
| **eta=4.0 (best)** | 3   | 68  | 645  | 46.4 | 95.17%    | 97.60% | 98.67% |
| eta=8.0            | 3   | 70  | 761  | 45.3 | 94.55%    | 97.38% | 98.62% |

**Finding.** `eta=4` beats the control on every metric. The win is concentrated in the TAIL:
mean required-P 70.5 -> 46.4 (-34%), p99 1042 -> 645 (-38%). Agreement at the deploy P=256 rises
+0.52pp vs control (+0.85pp vs shipped). `eta=2` already collapses the tail but loses a hair at
low P; `eta=8` regresses slightly past the sweet spot. Useful range eta in [4, 8], best ~4.

**Cost.** Zero at inference — identical `(D,K)` centroid matrix and the same stage-1 gemv. The whole
change lives offline in `build_clusters.py` (swap the assignment score, sweep one scalar).

**Why it works.** k-means minimises Euclidean error `||e-c||^2`; we rank by inner product `<h,e>`.
The error component that actually changes an inner product is the one PARALLEL to the embedding;
penalising it more (the `(eta-1)` term) reshapes clusters to preserve inner-product *ranking*, which
is exactly what stage-1 recall needs. The tail (tokens whose true cluster ranked ~1000th) is where
ranking was most distorted, so that's where the gain shows up.

**Runtime.** ~2.3 min per eta (model load + collect_hidden + 3 balanced passes over 152k rows).

## 2026-06-20 — finished the closed-form centroid (member mean vs ScaNN). Mean wins.

The first sweep used the member mean and flagged ScaNN's closed-form parallel-weighted centroid as a
"would only help more" upgrade. Built it (`centroids_scann`): the per-cluster D×D weighted
least-squares solve collapses to a cap×cap Woodbury solve because `A = nI + (eta-1)X̂ᵀX̂` is a
rank-cap update. Ran both centroid updates head to head at eta=4, same assignment rule, same budget.

| variant (eta=4) | p50 | p90 | p99 | mean | @128   | @256   | @512   |
|-----------------|-----|-----|-----|------|--------|--------|--------|
| member mean     | 3   | 68  | 645 | 46.4 | 95.17% | 97.60% | 98.67% |
| ScaNN closed    | 5   | 86  | 684 | 46.4 | 93.65% | 97.38% | 98.70% |

**Finding. The closed form does NOT help; the member mean is the operating-point optimum.** Same mean
required-P (46.4), but the closed form is worse at the deploy P (97.38 vs 97.60 @256, 93.65 vs 95.17
@128) and shifts p50/p90 up. Only @512 is a wash.

**Why.** Inference routes by COSINE — `Cn = C / ‖C‖` keeps only the centroid's *direction*. The
closed-form solve tunes magnitude and the parallel component as well, and normalising the centroid
throws exactly that part away. What survives is a direction slightly worse for top-1 recall than the
plain mean's. The closed form optimises the anisotropic *reconstruction loss*; we don't deploy that
loss, we deploy cosine recall, and the mean's direction already estimates the cluster's query
direction well. So the "conservative lower bound" framing was wrong: there is no closed-form upside to
capture here.

**Takeaway.** Promote the **member-mean** anisotropic partition as-is. Drop the closed-form centroid
from the promotion path. `centroids_scann` stays in the PoC as the recorded negative.

### Next steps if it graduates
- Re-run the full constrained-Lloyd from scratch (not warm-started) at eta=4 to remove the warm-start
  bias, and confirm the balance assert still holds exactly (K*cap==V).
- Stacking check: does it compose with always-score (the shipped lever) and with raising P? Both are
  orthogonal, so expected additive.
- Port into `surgery/build_clusters.py` behind an `--eta` flag (default 1.0 = current behaviour),
  rebuild a full artifact, and confirm the *end-to-end* agreement gain on the spliced model.
