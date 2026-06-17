# TurboHead — current state & next steps

CPU-only, ONNXRuntime. Work lives on branch `turbohead-fused-op`. **All measured numbers are in
[RESULTS.md](RESULTS.md)** (8-model speed×quality matrix + per-model reproduce commands); this file is
just the current state and what's left. Method writeup is in [../README.md](../README.md); the design
spec is [PLAN.md](PLAN.md).

## Where we are

The pipeline is implemented, reproducible, and committed:

- **Apply** (`surgery/`): `build_all.sh <hf-model> <slug>` builds the whole per-model artifact tree
  (int4 baseline → head weight → balanced clusters → 4 dense-head baselines → `onnx` + `fused` flash
  splices) into `artifacts/<slug>/`. Robust across architectures (RTN int4 body, Gemma rope shim,
  config copy, self-describing `hf_model_id.txt`).
- **Run** (`inference/decode_loop.py`): self-contained raw-ORT decode loop, greedy + probed-softmax
  sampling, contract A/H auto-detect, generic state handling — drives standard, hybrid (conv/SSM +
  sparse attention, e.g. LFM2.5), and embeds-in (Qwen3.5) models with no per-model config.
- **Fused kernel** (`csrc/turbohead_op.cc`): `FlashHeadSelect` collapses stage 2 into one pass emitting
  the candidate shortlist; the fastest backend.
- **Measure** (`eval/`): `turbohead-bench` (speed matrix, median±std, subprocess-isolated) and
  `turbohead-head-quality` (agreement + PPL on real hidden states).

## Headline finding

Fused FlashHead decodes **1.45×–5.37× faster than an fp32-equivalent dense head** @1 thread across the
8 models, at 94–98% greedy agreement. **The win scales with the head's share of a decode step** (narrow
hidden `D` + large vocab `V` + few/light layers ⇒ bigger win) — that's the whole story of the spread.
Caveats that stay true:

- Against an **int8** dense head (genai's default) the Amdahl ceiling is only ~1.33× (head ≈30% of the
  step). The big numbers are vs an **fp32** head, which is itself slower — both framings are in RESULTS.
- More threads *shrink* the speedup (head is memory-bound; cores also speed the dense baseline) →
  single-thread is the deploy point.
- Flash full-distribution PPL is coverage-limited; greedy/shortlist-sampling is where it shines. Read
  PPL within-model only; top-1 agreement is the cross-model metric.

## Open items

1. **Land the branch.** Merge `turbohead-fused-op` → `main` once reviewed.
2. **(Optional) coverage/speed Pareto** — small `(cap, P)` sweep to pick the knee instead of fixed
   `cap=16, P=256`; mainly helps big-vocab models (danube3's profile shows `P` barely moves small-vocab
   step time — see its P-sensitivity note in RESULTS).
3. **Self-contained embeds-in deploy** — `splice` could ship a fp16 `embed.npy` in the model dir so
   embeds-in models (Qwen3.5) don't fall back to `../head_W.npy`.
4. **Sampling** is temperature-only; add top-p/top-k filtering over the candidate shortlist if needed.
