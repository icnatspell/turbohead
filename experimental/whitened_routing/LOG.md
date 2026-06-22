# whitened_routing — run log

## 2026-06-20 — NEGATIVE result

Setup: fit mu, Sigma on 4000 train hidden states; fold A=(Sigma+lambda*tau*I)^-1 into the shipped
centroids; eval held-out required-P. Diagonal whitening (recall_lift's variant) included as a
reference. Lower required-P = better.

| routing                | p50  | p99  | mean   | @128   | @256   | @512   |
|------------------------|------|------|--------|--------|--------|--------|
| cosine (baseline)      | 5    | 1429 | 86.2   | 93.83% | 96.65% | 97.78% |
| diag-whiten            | 8    | 9495 | 884.6  | 74.17% | 78.12% | 81.83% |
| mahalanobis lam=1.0    | 151  | 9307 | 1646   | 48.90% | 54.10% | 59.88% |
| mahalanobis lam=0.1    | 759  | 9378 | 2309   | 33.00% | 38.22% | 45.35% |
| mahalanobis lam=0.01   | 1173 | 9494 | 2843   | 28.95% | 34.15% | 40.42% |

(col-normalising the folded centroids barely moves any of these.)

**Finding. Whitening loses, badly, and monotonically: the closer A gets to the true inverse
covariance (smaller lambda), the worse routing gets.** Cosine wins decisively.

**Why (the useful lesson).** Whitening de-emphasises high-variance hidden-state directions on the
assumption they are nuisance. In LLM hidden states they are the opposite -- they carry the signal
that separates clusters. Suppressing them throws away exactly the information routing needs. The
anisotropy of LLM hidden states is informative, not noise.

This is consistent with, and reinforces, `anisotropic_clustering`: the win there comes from shaping
the PARTITION to respect inner-product (parallel) structure, NOT from reshaping the query metric.
Keep the cosine router; put the data-awareness in the clustering objective instead.

**Verdict: parked.** A pure rotation is a no-op for single-codebook routing, and a learned non-
orthonormal metric hurts. OPQ-style rotation would only have teeth paired with a product quantizer
(see `factorized_router`); on its own there is nothing here.
