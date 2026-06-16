# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TurboHead replaces a quantized LLM's dense `[V,D]` language-model-head matmul with an approximate
clustering head (the FlashHead method, arXiv 2603.14591) and **splices it into an existing ONNX
model**, so any `onnxruntime` CPU deploy gets the speedup without a custom runtime. Target:
`Qwen/Qwen3-0.6B`, int4 body / int8 head baseline, CPU EP. README.md has the full method writeup;
`docs/PLAN.md` is the handoff spec (sections like §6 are referenced in code comments);
`docs/NEXT_STEPS.md` has current findings.

## Commands

```bash
uv sync                                    # install (full toolkit); `dev` group adds ruff/pytest/pyrefly
uv run ruff check turbohead/ tests/        # lint (CI gate)
uv run pytest -q                           # tests; integration tests self-skip without the model artifact
uv run pytest tests/test_select.py -q      # a single test file
```

The offline pipeline (run in order from repo root; each writes to `artifacts/`):

```bash
bash turbohead/surgery/convert_baseline.sh   # 1. int4/int8 baseline ONNX (genai model builder)
uv run turbohead-extract-head                # 2. dump bf16 head weight -> head_W.npy
uv run turbohead-build-clusters              # 3. balanced k-means -> clusters.npz  (--cap 16 | --clusters K)
uv run turbohead-splice -P 256 --stage1 int4 --block-size 128   # 4. splice -> qwen3_0_6b_flash/
```

Run / measure (console scripts map to `<module>:main`, see `[project.scripts]`):

```bash
uv run turbohead-decode artifacts/qwen3_0_6b_flash --reps 5 [--profile] [--temperature 0.8]
uv run turbohead-agreement --npz artifacts/clusters.npz   # top-1 vs dense argmax (sweeps P)
uv run turbohead-ppl       --npz artifacts/clusters.npz   # dense vs flash PPL + coverage
```

## Architecture

Three packages, split so the deploy path carries no offline-only deps:

- **`surgery/`** — offline, apply the method. `build_clusters.py` does balanced k-means (constrained
  Lloyd; exact `K·cap==V`, no padding) → `Cnorm (D,K)`, `Wperm (K,cap,D)`, `Vmap (K,cap)`.
  `build_subgraph.py::make_flash_nodes()` is the reusable op-chain builder, shared by the standalone
  verifier (`build_subgraph.py __main__`) and `splice.py`. `quantize_stage1()` rewrites just the
  stage-1 MatMul to `MatMulNBits` post-build.
- **`inference/decode_loop.py`** — the *only* file needed to run a spliced model. Self-contained:
  raw `onnxruntime` + numpy + tokenizer, manual KV cache, no genai/torch. Doubles as the profiler.
- **`eval/`** — dev quality gates (`agreement.py`, `ppl.py`); pure-numpy reference flash, hooks the HF
  model's `lm_head` input over WikiText-2.

### The flash head op chain (contract A, two stages)

1. **Stage 1 (coarse):** `sims = h · Cᵀ` over `K` centroids → `TopK(P)`. This `[K,D]` gemv at M=1 is the
   dominant head cost; it runs as int4 `MatMulNBits` (W4A16, `accuracy_level=4`) — ~9× faster than fp16
   at M=1 while holding 100% standalone argmax. **`MatMulNBits` is the only int gemv that wins at M=1**
   (it fuses dequant into the gemv); `MatMulInteger`/manual dequant were tried and lost.
2. **Stage 2 (refine):** `Gather` the `P·cap` candidate rows, dot with `h` for exact logits, scatter
   into a `-1e9`-filled `(1,V)` base. Stage 2 stays **fp32** — the CPU EP has no fp16 matmul kernel
   (it auto-casts), so storing fp32 skips an implicit fp16→fp32 cast of the gathered rows.

### Head contracts (auto-detected from graph outputs in `decode_loop.py`)

- **A (logits-out):** graph emits `logits` `(1,1,V)`; argmax/sampling done in Python. *Current spliced graph.*
- **B (token-out):** graph emits the next-token id directly; V never materialized. Greedy-only.

Sampling (`temperature>0`, contract A only) does a probed-softmax over **only** the scored candidates,
skipping the full-vocab softmax the dense head pays — so sampling speeds up more than greedy.

## Key facts to keep in mind

- **Amdahl ceiling ~1.33×:** the head is ~30% of a decode step, so even a free head caps speedup there.
  Measured: ~1.21× greedy / ~1.35× sampling at 1 thread. More threads *shrink* the gain (head is
  memory-bound; extra cores also speed the dense baseline). Best single-threaded.
- **Coverage caps full-distribution PPL.** A true token outside the probed set gets ≈0 probability.
  Raise `-P` to lift coverage (and accuracy), at a speed cost. Read agreement and coverage together.
- `cap` must divide `V` exactly (`V=151936=2⁷·1187`). Downstream steps read `cap`/`K` from the `.npz`;
  only `-P`/`--stage1`/`--block-size` shape inference.
- `onnx.checker` is deliberately skipped on save — it rejects genai's `com.microsoft` contrib ops; ORT
  load validates instead. The dense head node is removed but its weight initializer stays (tied embed).
- IOBinding zero-copy KV is a net loss under contract A (must pull `logits` to numpy anyway; reusing
  ORT output buffers as next-step inputs segfaults). See the `Decoder` docstring.
- Code comments mark deliberate simplifications with `ponytail:` and reference `docs/PLAN.md` sections (§N).
