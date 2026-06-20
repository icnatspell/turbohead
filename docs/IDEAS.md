# Ideas to improve FlashHead

Directions worth exploring, grouped by which part of the method they attack. Written so a junior
ML engineer with no prior exposure can follow it: shared terms are defined once below, and each
idea explains its own jargon. For more background, `RELATED.md` explains how FlashHead compares to
other methods, `THESIS_ADAPTIVE_PROBING.md` covers the two stages in depth, and
`DIFFERENCE_OURS_VS_FLASHEAD.md` covers the clustering internals.

## Background: what FlashHead does, in two stages

A language model ends with a **head**: it takes the model's hidden state `h` (a vector of length
`D`) and produces one score, a **logit**, for each of the `V` vocabulary tokens. A token's logit is
just its embedding row dotted with `h`. Normally this means one big matrix-vector multiply (a
**gemv**) over all `V` tokens, which is slow because `V` is large (about 152,000 for Qwen3-0.6B).

FlashHead avoids scoring all `V` tokens by grouping them into `K` clusters offline (`K=9496` for
Qwen3-0.6B), each holding `cap` tokens (`cap=16`). At decode time it runs two stages:

- **Stage 1 (routing):** score `h` against all `K` cluster centroids (`sims = C·h`) and keep the
  top `P` clusters. This always scans all `K` centroids, so its cost does not depend on `P`. It is
  the floor on how fast the head can get.
- **Stage 2 (refine):** take the `P·cap` tokens inside those `P` clusters, compute their exact
  logits, and return the best one. Its cost grows with `P`.

Most prior work (adaptive probing) shrinks stage 2 by using a smaller `P`. The most under-explored
lever is stage 1.

## Terms used throughout

- **top-1 / agreement:** did the fast method pick the *same single best token* as the exact
  full-`V` head? This is the headline quality metric. "agree@256" means agreement when `P=256`.
- **required-P:** for a given token, the smallest `P` that would have included its true cluster in
  the top-`P` routed clusters. If the true cluster ranks 5th among the `K` centroids, required-P is
  5. Smaller is easier. Because stage 2 is exact, FlashHead gets the token right exactly when `P`
  is at least its required-P.
- **percentiles (p50, p90, p99):** the value below which that fraction of tokens fall. p50 (the
  median) of required-P being 5 means half the tokens need `P` of 5 or less; p99 of 1165 means the
  hardest 1% need `P` above 1165. A big gap between p50 and p99 is a "heavy tail": most tokens easy,
  a few very hard.
- **cosine vs inner product:** stage 1 ranks clusters by *cosine* similarity, which compares only
  the *direction* of two vectors (it normalizes away their length). The true target is the largest
  *inner product* (raw dot, which keeps length). Searching for the largest inner product is called
  **MIPS** (maximum inner product search).
- **perplexity (PPL):** a standard measure of how well the model predicts held-out text. It is the
  exponential of the average negative log-probability the model assigns to the true next tokens.
  Lower is better; assigning near-zero probability to a true token sends PPL toward infinity.
- **softmax, Z, numerator:** to turn logits into probabilities you exponentiate each and divide by
  their sum. That sum is the denominator `Z` (the "partition function"); the chosen token's
  exponentiated logit is the numerator. If you only score some tokens, `Z` is wrong (too small) and
  unscored tokens get probability ~0.
- **nat:** the unit of log-probability when using natural log. A gap of 11 nats means one
  probability is `e^11` (about 60,000x) larger than another.

## Attacks the stage-1 floor

### 1. Hierarchical / coarse-to-fine routing

Today stage 1 is one flat matmul over all `K` centroids. That cost grows as you add clusters, and
the paper scales `K` into the tens of thousands. The idea: cluster the centroids *themselves* into
about `√K` "super-centroids", then route in two cheap steps. First score the super-centroids,
descend into the winning few, then score only the leaf centroids underneath them. Stage 1 work
drops from roughly `K·D` to roughly `√K·D`. (This is the "inverted-multi-index" trick from
nearest-neighbor search, applied one level up.) It is the only lever that lowers the stage-1 floor,
which adaptive probing cannot touch, and it matters most exactly where the paper is heading (large
`K`).

**POC result: tested, not worth it in the naive form.** See `experimental/hierarchical_stage1/hierarchical_stage1_poc.py`.
On Qwen3-0.6B (4000 WikiText-2 positions, flat baseline agreement at P=256 is 96.8%):

- Single-assignment 2-level routing (each leaf belongs to exactly one super-centroid): `M=64`
  super-centroids, probing `m=8` of them, cuts stage 1 by about 13x but agreement drops to 59%.
- Soft assignment (each leaf belongs to its top-`r` super-centroids) helps but not enough. The
  best operating point, `M=100`, `r=3`, `m=16`, gets 4.6x reduction at 75% agreement.

Two findings explain why:

1. **Recall is the wall.** "Reachable" means the true cluster's super-centroid made the top-`m`, so
   the true cluster is still in play after the coarse prune. Across every setting `agree@256 ≈
   reachable`: once the true super-centroid survives, the leaf ranking succeeds; the coarse prune
   itself is what drops the true cluster. Pushing reachability near 100% needs probing so many
   super-centroids that the savings vanish.
2. **The heavy tail that makes adaptive probing attractive is what breaks this.** The true cluster
   often sits deep in the flat order (p90=80, p99=1165, from the adaptive-probing analysis). A
   coarse first level cannot keep a flat-rank-500 cluster in its top few super-centroids, so it
   prunes exactly the hard tokens.

On top of that, stage 1 on Qwen3-0.6B is about 0.35 ms, only 1.6% of a decode step, so even a
working 30x reduction would be near-invisible end-to-end. Hierarchical stage-1 only pays off at
very large `K`, and only if the coarse level is made nearly lossless (a learned coarse quantizer,
or super-centroids fit to maximize reachability rather than plain cosine k-means). High bar, small
payoff. Parked.

### 2. Product-quantized centroids for stage 1

Stage 1 is limited by reading the `K·D` centroid matrix from memory. **Product quantization**
compresses each vector by splitting it into chunks and replacing each chunk with the id of a nearby
entry in a small prototype table, so the routing read shrinks. Alternatively, low-rank the centroid
matrix. Smaller win than hierarchical, but it stacks with it and with the int4-quantized stage 1
already in use.

## Cheap to test, could lower required-P directly

### 3. MIPS-aware cluster ranking (a one-array swap)

Stage 1 ranks clusters by cosine similarity (normalized centroid dotted with `h`). The actual
target is the cluster containing `argmax(e·h)`, a raw inner product. Normalizing the centroid
throws away the embedding's length, and in MIPS that length is part of what signals a cluster might
hold a high-scoring token. So test ranking by the unnormalized mean embedding, or by the mean plus
a per-cluster correction based on its radius or its largest-norm member. This is the same shape as
the data-aware routing prototype: swap the routing matrix, recompute required-P on the same hidden
states. Fast to falsify and grounded in standard MIPS theory.

**POC result: tested, no win.** See `experimental/mips_routing/mips_routing_poc.py`. Same 4000 WikiText-2 positions,
cosine baseline top-1 agreement at P=256 is 96.8%. Three routing swaps, with the cluster groups
held fixed (only the ranking score changes). Lower required-P is better:

| routing | p50 | p99 | mean rank | agree@256 |
|---|---|---|---|---|
| cosine (baseline) | 5 | 1165 | 75.7 | **96.8%** |
| raw mean (`mu·h`, unnormalized) | 4 | 1110 | 71.8 | 95.8% |
| `mu·h + ‖h‖·radius` (MIPS bound) | 4268 | 8824 | 4107 | 13.2% |
| `mu·h + ‖h‖·maxnorm` | 4005 | 9177 | 4110 | 11.6% |

The norm-bound variants are catastrophic: the radius term (`‖h‖·radius`, an upper bound on how high
a cluster member could score) swamps the actual `mu·h` term, so they end up sorting clusters by
radius and ignoring `h`. A small-coefficient sweep (`mu·h + a·‖h‖·radius`) finds a marginal tail
gain at `a=0.01` (p99 drops 1165 to 977, mean 75.7 to 68.6) but agree@256 still tops out at 96.2%,
under the cosine baseline.

Why cosine wins: the equal-size balanced clusters have similar radii, so the norm/radius MIPS
signal carries little distinguishing information and only adds noise. The embeddings sit near a
shared length scale, so cosine is already close to inner product for ranking, minus the harmful
length noise. The inner-product framing nudges the median (p50 4 vs 5) but loses the metric that
matters (top-1). Parked.

**Follow-up POC: three more free routing swaps, all lose.** See `experimental/recall_lift/recall_lift_poc.py` (lever 1),
12k WikiText-2 positions, held-out fit/eval split, cosine baseline agree@256 = 97.5%:

| routing swap | agree@256 | why it fails |
|---|---|---|
| learned-`h` prototype (mean of the hidden states that select each cluster) | 89.9% | `K=9496` clusters but only ~6k calibration tokens, so most clusters are fit from 0–1 noisy samples; the embedding-mean centroid uses all 16 members and is far more stable |
| mean-centered cosine (subtract the train-mean `h̄`) | 74.4% | the "common-mode" direction the hidden states share is *highly* discriminative for routing here, not removable noise — deleting it destroys the signal |
| diagonal-whitened cosine (scale dims by `1/std(h)`) | 77.3% | same: the anisotropy-removal trick from embedding-similarity work does not transfer, because the centroids live in the same anisotropic space and rely on that structure |

Conclusion across both POCs: **cosine-on-the-embedding-mean is at the ceiling for a single free
linear routing pass.** Stop hunting for a better routing *matrix*; the remaining wins are elsewhere.

## Improves fidelity, not speed

### 4. Coverage-corrected probabilities

Today a true token sitting in a cluster FlashHead did not probe gets a logit of `-1e9`, so its
probability is about zero. That caps perplexity (assigning ~0 probability to a real token blows up
PPL) and distorts sampling. The fix: each stage-1 centroid score is a stand-in for its cluster's
total probability mass, so add an analytic correction for the unprobed clusters to keep the softmax
denominator `Z` honest, without scoring all `V` tokens. This makes likelihood and sampling
first-class instead of needing the Monte-Carlo workaround the paper uses.

**POC result: tested, clear win on the part that matters.** See `experimental/coverage_correction/coverage_correction_poc.py`.
Qwen3-0.6B, 1000 WikiText-2 positions, with the genuine corpus next-token as the target (not the
model's own argmax, which would almost always be probed and hide the problem), `P=256`. At `P=256`,
11.2% of real next-tokens land in an unprobed cluster, so the hole is material.

| method | PPL | unprobed-only PPL |
|---|---|---|
| gold (full-`V` softmax) | 11.1 | 1640 |
| truncated (current FlashHead) | **10277** | ~1e30 |
| corrected `Z` + cluster-mean numerator (fully deployable, token unknown) | 33.1 | 5.6e7 |
| corrected `Z` + exact single-token logit (token known) | **9.76** | 1026 |

Two findings:

1. **The denominator fix is the real, cheap, robust win.** Add `cap · Σ exp(mean-logit_k)` over the
   unprobed clusters to the softmax denominator `Z`, where a cluster's mean-logit is `mu_k · h`
   (its mean embedding dotted with `h`). That costs one extra stage-1-sized gemv, or nothing extra
   if you store a per-cluster norm and reuse the stage-1 cosine scores. With that corrected `Z` and
   the true token's *exact* logit, PPL is 9.76, matching gold's 11.1. The current method's 10277
   was unusable for likelihood; this makes likelihood first-class.

2. **Estimating an *unknown* unprobed token's own logit from cluster aggregates fails.** Using the
   cluster mean (33.1) or the cosine score times a typical member norm (33.0) both underestimate by
   about 11 nats: a token the corpus actually chose sits far above its cluster's average member. A
   single global offset can match the PPL *number*, but only as an averaging artifact (individual
   per-token probabilities stay biased). So for free-running sampling, use the corrected `Z` (which
   fixes the over-confidence on the probed tokens) and sample the unprobed tail by cluster mean
   mass; do not trust individual unprobed-token probabilities.

**Where this pays off directly:** any path where the token is known. That includes PPL/likelihood
evaluation, and the speculative-decoding acceptance test (#5 below, and `RELATED.md`, needs
`P(drafted token)` for specific known tokens). Score those exact logits and add the corrected `Z`:
calibrated, gold-quality, with no full-`V` softmax. This unblocks the speculative-decoding
composition.

**Overhead applies only when the head emits probabilities** (sampling, PPL, the spec-decode
acceptance test). Greedy argmax decoding, the fused default, never builds a softmax, so the tail
term is never computed and the correction costs nothing there. When probabilities are needed, the
added work is the unprobed-cluster tail (`Σ exp(mean-logit)` over the `K` clusters), which is one
extra stage-1-sized gemv (`mu·h`) plus an `O(K)` exponentiate-and-sum, or just the `O(K)`
exponentiate-and-sum if the per-cluster norms are stored and the stage-1 cosine scores are reused.

## Concrete realization of adaptive probing

### 5. Cascade probing

Instead of predicting the right `P` up front, probe a small `P` first, check the confidence gap
between the top two refined logits, and re-probe with more clusters only when that gap is small.
Stage 1 already ran, so the second pass is just extra gathers in stage 2. This sidesteps the hard
"can we predict required-P" question by measuring uncertainty after a cheap first look. A low-risk
variant of the adaptive-probing thesis.

**POC result: tested, ~break-even with just raising `P`.** See `experimental/recall_lift/recall_lift_poc.py` (lever 3),
12k WikiText-2 positions. Probe `P0=64`, escalate to `P1=512` when the gap between the top-two
*probed* exact logits is below a threshold. At threshold 2.0: 54.8% escalate, average `P=309`,
agree 97.6% — essentially the same as fixed `P=256` (97.5%) for the same cost. The margin is a weak
predictor because the failure mode is a winner you *didn't* probe, which the margin over the probed
set cannot see. Confirms the "no cheap confidence signal" open problem above.

We also tested the **exact-stop** variant (lever 2): probe in cosine order, stop the moment a
provable upper bound (`mu_k·h + ‖h‖·radius_k`, Cauchy-Schwarz) says no unprobed cluster can beat the
best logit found, giving a *guaranteed* exact top-1. It degenerates: average stop-`P` ≈ `K` (the
whole vocab). The bound is far looser than the gaps between logits, so almost every cluster "could"
still hold the winner — certifying the max costs as much as the dense head. Parked.

### 8. Always-score the tokens FlashHead most often misses  ✅ implemented

The other ideas chase the heavy tail in general. This one asks a narrower question: **are the misses
the *same* tokens every time?** They partly are. The graph already always-scores a few special
tokens (EOS/BOS) by stuffing their weight rows into the stage-2 `Wspec`/`spec_ids` path, scored
every step regardless of routing. Lever 4 extends that list with the tokens a calibration pass found
FlashHead routes badly — so they can never be missed, at the cost of a handful of extra scored rows.

**POC result: the win.** See `experimental/recall_lift/recall_lift_poc.py` (lever 4), 12k positions, held-out. Fitting
the most-missed set on the train half and measuring on eval, a **64-token** always-score list
rescues **52% of all misses → agree@256 rises 97.5% → 98.8%**. The curve plateaus immediately
(top-64 = top-4096): about half the misses are a small set of frequent tokens (function words,
punctuation) that route badly *independent of context*; the other half are idiosyncratic one-offs no
fixed list can catch. The cost is ~free — the always-scored rows ride the existing `Wspec` path, a
few dozen extra dot products per step against a ~17 MB-per-step head.

Why it works where routing tricks don't: a chronically-misrouted frequent token has a bad *cluster
assignment* (its embedding sits far from its cluster's centroid), so no stage-1 score will rank it
in. Routing can't fix a bad assignment; always-scoring sidesteps routing for that token entirely.

It **stacks** with simply raising `P` (`P=256→512` independently buys 97.5%→98.3%), so
`always-score + P=512` reaches the ~99% region at modest cost.

How to build/use it (this is what shipped):

- **Calibrate** (offline, needs the HF model): `turbohead-calibrate-misses --model <hf> --npz
  <clusters.npz> -P 256 --top 64 --out <dir>/always_score.npy`. One pass over WikiText-2: capture the
  hidden states, find the dense argmax tokens whose cluster ranks below `P`, write the `--top` most
  frequent ids. It is calibration-data- and model-specific (refit per model).
- **Splice it in**: `turbohead-splice ... --always-score <dir>/always_score.npy`. Those ids are
  unioned into the always-scored specials. **Omit the flag to turn lever 4 off** — that is the
  on/off switch. `build_all.sh` runs the calibration and passes the flag automatically; set
  `ALWAYS_SCORE=0` to skip it.

## Beyond the head: the KV cache (body, orthogonal to FlashHead)

These attack why tokens-per-second *drops as the context grows*, which is a body problem the head
cannot touch. Decode per-step latency on Qwen3-0.6B is roughly `19 ms + 0.05 ms × S` (S = context
length): a fixed floor (the int4 body matmuls) plus a term that grows linearly with how many tokens
are already in the cache. That linear term is the **KV cache** (the stored keys/values every new
token attends to). It splits in two: the *fundamental* part is attention reading the whole cache
once per step; the *avoidable* part is copying the cache in and out each step.

### 6. Buffer-shared KV cache (DONE, positive — shipped in `decode_loop.py`)

The default decode loop fed the cache back as numpy every step, so ONNX Runtime reallocated and
copied the entire cache (about 224 KB per token) on every token. `share_kv` instead pre-allocates
one fixed max-length buffer per layer and tells the attention op to write each new key/value
*in place* (bind past-input and present-output to the same memory; the attention op learns the
valid length from a full-width attention mask). No reallocation, no copy.

**Result: byte-identical output, the per-step growth slope roughly halves** (~53 → ~23 ms per 1000
context tokens), giving ~1.2× decode at 400 tokens and more as context grows. Measured against
onnxruntime-genai's own runtime on the same graph, which hits the same slope, so this is the
expected ceiling for the copy fix. POC and A/Bs in `experimental/buffershare/buffershare_poc.py`, `experimental/genai_kv/genai_kv_poc.py`,
`experimental/kv_scaling/kv_scaling_poc.py`.

**Applicability.** On by default and verified byte-identical on the 6 pure-KV transformers (Qwen3
0.6B/1.7B, Gemma3 270M/1B, Llama-3.2-1B, danube3-500M). Auto-disabled — falls back to the
numpy path, no regression — on the two models with fixed-size state (LFM2.5 hybrid conv/SSM,
Qwen3.5), since the in-place trick needs every state tensor to grow by one row per step. The gate
(`all state inputs grow` and `attention_mask` exists) fails closed: a model that doesn't qualify is
slower, never wrong.

**Limit: the buffer is finite, and we stop at it.** The buffer is pre-allocated to a fixed length
(`max_kv`, default = exactly this request's `prompt + max_new`). The buffer length is also the memory
ceiling: about 224 KB/token on Qwen3-0.6B, so a 2048 buffer is ~450 MB, a 32k buffer ~7 GB — that
linear up-front cost is inherent to static KV, the same trade genai makes. **When a generation would
exceed the buffer we clamp `max_new` and stop early; we do not drop old tokens.** We deliberately do
*not* fall through to writing past the buffer: ORT's GQA writes each new key/value in place at offset
= past-sequence-length, so an over-long write lands outside the allocated arena — an out-of-bounds
write, i.e. segfault or silent corruption, not a graceful realloc (static KV has no growth path). The
clamp is what turns that into a clean, deterministic short return.

**To overcome it (sliding window / StreamingLLM), and why it is not a one-liner.** Generating past
the cap without growing memory means evicting the oldest tokens. The hard part is that ORT's GQA
derives the RoPE position, the write offset, and `seqlens_k` all from the *same* past-length value —
you cannot ask it to "write at slot 1792 but rotate as if at absolute position 9000". So a plain ring
buffer is impossible; the only correct shape is compaction: (1) left-shift every KV buffer to drop a
chunk of oldest tokens (a copy, but rare if you drop in big chunks — amortized cheap), then (2)
**re-rotate all surviving cached keys** into the compacted position frame, because the old keys were
rotated at their original absolute positions and the next new key would otherwise be rotated at a
small offset, breaking relative distances. Step (2) must replicate the model's exact RoPE (theta
base, partial-rotary fraction, any scaling) per model, in numpy, and the result is no longer
byte-identical to the reference — it is lossy by construction (dropped context changes output), so the
clean A/B regression test stops being the safety net. Realistic size ~40–80 lines plus per-model RoPE
care. Build it only for an actual long-running-chat workload that hits the cap; until then,
`max_kv`-bounded early-stop is the correct, predictable default. (`local_window_size` on the GQA op
only masks attention scope — it does *not* bound the cache memory, so it does not solve this.)

### 7. Quantized (int8 / int4) KV cache

After buffer-sharing, the part of the slope that remains is attention *reading* the fp32 cache. Store
the cache in int8 or int4 instead of fp32 and that read moves 4× or 8× fewer bytes, shrinking the
residual slope. *What it is:* the same idea as int4 weights (`MatMulNBits`) applied to the keys and
values, dequantized on the fly inside attention.

The blocker is purely kernel support: ONNX Runtime's CPU `GroupQueryAttention` only handles
fp32/fp16 cache today, and `-p fp16 -e cpu` from the genai builder drops both the int4 body and the
GQA op and then crashes on CPU. So this is not a re-export — it is a **custom attention op** with a
quantized cache, the same class of effort as the FlashHead op, with the extra wrinkle that the cache
read is strided rather than one dense matmul. High ceiling on long-context throughput, real build
cost. Park behind shipping #6 broadly.

## Not worth it

- **Variable cluster size by token importance.** Fights the equal-size kernel requirement (the
  inference kernel gathers rows with plain arithmetic only because every cluster has the same `cap`
  tokens), and the paper's own ablation (their Table 6) shows equal clusters beat unequal ones.
- **Caching cluster rankings across decode steps.** The hidden state `h` moves too much from one
  token to the next for reusing a previous step's ranking to pay off.

## Suggested order

1. Raise `P` if more agreement is wanted (the only reliable knob on the idiosyncratic tail);
   stacks with #8. `P=256→512` buys 97.5%→98.3% at ~+7% decode latency.

Done, positive: always-score frequent misses (#8), shipped — `turbohead-calibrate-misses` +
`turbohead-splice --always-score`, +1.3pp agreement (97.5%→98.8%) at ~free cost, on/off via the flag.
Parked: free routing swaps (#3) and cascade/exact-stop (#5), both no better than fixed `P` — see
`experimental/recall_lift/recall_lift_poc.py`.
Done, positive: buffer-shared KV (#6), shipped — halves the per-step-vs-length slope, byte-identical.
Coverage correction (#4). Corrected-`Z` likelihood matches gold PPL (10277 down to
9.76) when the token is known, which unblocks PPL evaluation and the spec-decode acceptance test.
Promote to implementation: store the per-cluster mean-logit norms in the npz, and add the tail term
where the head emits probabilities.

Next on the body axis: quantized KV cache (#7), a custom attention op — the high-ceiling lever for
long-context throughput once buffer-sharing is in everywhere.

Parked: MIPS-aware ranking (#3), because the POC showed cosine already beats every inner-product
and norm-bound routing on top-1. Hierarchical stage-1 (#1), because the POC showed it is
recall-bound and only relevant at very large `K`. Product-quantized centroids (#2), a small win
that is only useful stacked on a working #1.

Related external methods (graph indices, learned routing, and others) and how they compare live in
`RELATED.md`.
