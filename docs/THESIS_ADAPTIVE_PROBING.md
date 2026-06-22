# Thesis avenue: adaptive probing for FlashHead

A proposal for a master's thesis built on this repo. Written for a junior engineer:
it explains the background, the measured evidence, how to reproduce it, and what to build.

## Background you need first

FlashHead replaces a language model's output head (the big `[V, D]` matrix that turns a
hidden state into one logit per vocabulary token) with a two-stage retrieval:

1. **Stage 1 (routing).** Score `K` cluster centroids against the hidden state `h`
   (`sims = C · h`), then take the top `P` clusters. `P` is the **number of probes**.
2. **Stage 2 (refine).** Gather the tokens in those `P` clusters (`P · cap` of them,
   where `cap = V / K` is the cluster size), compute their exact logits, pick the best.

The paper fixes `P` at one value for every token (it uses `P = 512`). Stage 2 reads
`P · cap · D` weight bytes per token, so its cost grows with `P`. Stage 1 always scores
all `K` centroids, so its cost does not depend on `P`.

**The question this thesis asks:** does every token need the same `P`? If most tokens are
easy and only a few are hard, you could spend a small `P` on the easy ones, a large `P` on
the hard ones, and get the same accuracy for far less work.

## The core idea

Pick `P` per token instead of using one fixed `P`. Use a cheap signal from stage 1 (for
example, how peaked the centroid scores are) to decide: confident routing gets few probes,
uncertain routing gets many.

This only pays off if two things hold. First, the per-token difficulty must vary a lot.
Second, you must be able to predict that difficulty cheaply. We measured the first. The
second is the open research question.

## What "difficulty" means here, precisely

Stage 2 is exact over the tokens it gathers. So FlashHead returns the correct top-1 token
**exactly when the cluster holding that token is among the top `P` probed clusters**. Define
a token's **required P** as the rank of its true cluster in the stage-1 score ordering. Rank
1 means the right cluster scored highest (one probe is enough). Rank 200 means you need
`P ≥ 200`.

You can compute required-P directly, without rerunning FlashHead at every `P`:

1. Run the real model on text, capture the hidden states `h` going into the head, and record
   the true next token (the dense head's argmax).
2. For each token, compute `sims = C · h`, find the cluster that contains the true token, and
   count how many clusters scored higher than it. That count (plus one) is the required P.

The distribution of required-P across many tokens is the **oracle ceiling**: the best any
adaptive scheme could do if it knew each token's difficulty for free.

## The measured evidence (Qwen3-0.6B, 4000 WikiText-2 positions)

Required-P is heavy-tailed. Half the tokens need 5 probes or fewer. The hardest 1% need
more than 1165.

```
required-P percentiles:  p50=5   p90=81   p95=155   p99=1165   p99.9=9416   (K=9496)
```

That shape is exactly what adaptive probing wants: a cheap common case and a rare expensive
tail. The payoff table below pairs each fixed `P` with the oracle's average `P` that reaches
the **same accuracy**, then converts to speedup.

| fixed P | top-1 accuracy | oracle avg P | stage-2 speedup | end-to-end (CPU, 0.6B) |
|--------:|---------------:|-------------:|----------------:|-----------------------:|
|      64 |          87.4% |         17.1 |            2.3× |                 1.009× |
|     128 |          93.7% |         22.8 |            3.5× |                 1.020× |
|     256 |          96.8% |         28.3 |            5.9× |                 1.04×  |
|     512 |          97.9% |         34.9 |            9.9× |                 1.09×  |

Read the `P=256` row: a perfect oracle matches 96.8% accuracy while averaging 28 probes
instead of 256. That is 5.9× less stage-2 work.

### Why end-to-end barely moves here, and where it would

On this machine (Qwen3-0.6B, CPU, one thread) the whole head is about 7% of a decode step,
and stage 2 is about 5%. Shrinking 5% by 5.9× saves almost nothing end-to-end. The body
matmuls dominate (about 74%).

The stage-2 win is real; the end-to-end translation depends on how big the head is. Holding
the same oracle (`P=256` → avg 28.3), the end-to-end speedup scales with head share:

| head share of step | end-to-end speedup |
|-------------------:|-------------------:|
|  7% (this CPU, 0.6B) | 1.06× |
| 20% (larger model / int4 GPU) | 1.20× |
| 40% (Gemma3-270M, head-heavy) | 1.50× |
| 60% (paper's stated upper bound) | 2.0× |

**Evaluate this thesis on head-heavy models and on GPU**, where stage 2 is a real fraction
of the step. On a narrow-`D`, large-`V`, few-layer model like Gemma3-270M, the head is a big
slice and the win shows up.

## The two caveats that shape the thesis

1. **The numbers above are an oracle.** The oracle knows each token's required-P for free. A
   real predictor sees only stage-1 outputs (the `sims` vector) and must guess. The thesis is
   how much of the 5.9× a practical predictor keeps. Capturing half of it would already be a
   strong result.

2. **Adaptive `P` shrinks stage 2 only.** Stage 1 scores all `K` centroids no matter what, so
   it becomes the floor on head speedup. This is another reason the win is clearest on
   models where stage 2, not stage 1, dominates the head.

## What a thesis would build and measure

1. **Confirm the ceiling on more models.** Run the headroom analysis on Gemma3-270M,
   Llama-3.2-1B, and a large-vocab model. Report required-P distributions and oracle tables.

2. **Design a predictor.** Map stage-1 outputs to a probe count. Candidate signals:
   - entropy of `softmax(sims)` (flat means uncertain means more probes),
   - the gap between the top centroid score and the runner-up,
   - the top-`m` score mass.
   Fit a simple rule or a tiny calibrated model that outputs `P` for a target accuracy.

3. **Measure realized speedup vs the oracle.** For each predictor, report accuracy, average
   `P`, and measured latency against the fixed-`P` baseline at matched accuracy. The headline
   metric is "fraction of oracle speedup captured."

4. **Implement it in the kernel.** The current fused op takes a fixed-length probe list. An
   adaptive version needs a variable-length probe list, or a fixed buffer with an early stop.
   On GPU this means a dynamic grid; on CPU it means a variable loop bound. Measure the real
   cost, including any overhead the variable length adds.

5. **Stretch goal: spend the budget, do not just cut it.** Instead of cutting average `P` at
   fixed accuracy, hold a fixed average `P` and route the saved probes to the hard tail. This
   raises accuracy and coverage for the same cost. Coverage matters for perplexity and for
   sampling, so this connects to full-distribution fidelity.

## How to reproduce the headroom analysis

The throwaway script lives at `logs/adaptive_probe_headroom.py`. It captures hidden states
from the real model and prints the required-P distribution plus the oracle payoff table.

```bash
# default: Qwen3-0.6B against artifacts/qwen3_0_6b/clusters.npz
uv run python logs/adaptive_probe_headroom.py
# another model (edit the npz path in the script to match the slug)
uv run python logs/adaptive_probe_headroom.py google/gemma-3-270m
```

To measure stage-2 latency vs `P` (the timing model behind the speedup columns), splice the
fused head at several `P` values and read the `FlashHeadSelect` line from the profiler:

```bash
R=artifacts/qwen3_0_6b
for P in 32 256 512; do
  uv run turbohead-splice --backend fused --src $R/baseline --dst $R/fused_P$P \
    --npz $R/clusters.npz --head $R/head_W.npy -P $P
  uv run turbohead-decode $R/fused_P$P --reps 3 --profile
done
```

Measured points on this machine (one thread): `FlashHeadSelect` was 0.18 ms at `P=32`,
1.55 ms at `P=256`, 2.06 ms at `P=512`. Stage-2 grows about linearly with `P`. The rest of
the step held near 22.1 ms.

## Two adjacent directions worth a mention

- **Data-aware clustering.** The hard tail is rarer tokens. In the same run, the token the
  model wanted had a median corpus frequency of 45 for easy positions (rank ≤ 8) and 8 for
  hard positions (rank > 64). Standard spherical k-means clusters token embeddings without
  knowing which tokens the hidden states actually reach. A clustering that uses real hidden
  states, or that always scores the most frequent tokens, could pull rare-but-needed tokens
  into better-ranked clusters. This shrinks the tail that forces large `P`, so it stacks with
  adaptive probing.

- **Speculative decoding (for example EAGLE-3).** Speculative decoding attacks the body: a
  draft proposes several tokens, the target verifies them in one pass, which amortizes the
  body's weight reads. FlashHead attacks the head. They compose. During verification the
  target head runs on every drafted position, so per accepted token the head runs more often
  relative to the body. Speculation pushes the bottleneck back toward the head, which is
  FlashHead's target. EAGLE-3's reduced draft vocabulary (around 32k) lives on the draft side
  and leaves the target's verification head untouched, so it does not replace FlashHead. The
  open question is the acceptance test: it needs the target's probability for each drafted
  token, which may sit outside FlashHead's probed set. You can score those specific tokens
  directly (the same trick the fused op uses for always-scored special tokens), so the
  research question is whether the resulting acceptance rate matches a full-vocab target.
