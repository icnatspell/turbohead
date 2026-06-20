# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TurboHead replaces a quantized LLM's dense `[V,D]` language-model-head matmul with an approximate
clustering head (the FlashHead method, arXiv 2603.14591) and **splices it into an existing ONNX
model**, so any `onnxruntime` CPU deploy gets the speedup. CPU EP, int4 body. Two flash backends:
`onnx` (portable standard ops, logits-out) and `fused` (a custom CPU op, shortlist-out — fastest).
Swept across **8 models** (Qwen3, Gemma3, Llama-3.2, LFM2.5, h2o-danube3, incl. hybrid conv/SSM +
attention archs): fused decodes **1.45×–5.37× faster than an fp32-equivalent dense head** @1t, scaling
with the head's share of a decode step. Reference config: `Qwen/Qwen3-0.6B`. README.md has the method
writeup + repo map; **`docs/RESULTS.md` has all numbers, key findings, reproduce commands, and open
items**; `docs/ORT_QUIRKS.md` records the measured ORT CPU quirks behind the precision/op choices.

## Commands

```bash
uv sync                                    # install (full toolkit); `dev` group adds ruff/pytest/pyrefly
uv run ruff check turbohead/ tests/        # lint (CI gate)
uv run pytest -q                           # tests; integration tests self-skip without the model artifact
uv run pytest tests/test_select.py -q      # a single test file
```

The offline pipeline is **one command** per model → `artifacts/<slug>/` (self-contained, idempotent;
tees logs; `FORCE=1` rebuilds). It runs: int4 baseline → `head_W.npy` → balanced clusters → 4 dense
baselines (`turbohead-quantize-head`) → `onnx` + `fused` splices, then prints the bench/eval commands.

```bash
bash turbohead/surgery/build_all.sh Qwen/Qwen3-0.6B qwen3_0_6b   # -> artifacts/qwen3_0_6b/{baseline,head_W.npy,clusters.npz,head{16,8g128,4g128,4g32},onnx,fused,logs}
```

Underlying steps (what build_all wires per-slug; console scripts map to `<module>:main`):
`convert_baseline.sh` (MODEL/OUT env), `turbohead-extract-head`, `turbohead-build-clusters`
(`--cap 16 | --clusters K`), `turbohead-splice --backend {fused,onnx} --src/--npz/--head -P 256`.

Run / measure:

```bash
R=artifacts/qwen3_0_6b
uv run turbohead-decode $R/fused --reps 5 [--profile] [--temperature 0.8]    # raw-ORT deploy loop
uv run turbohead-bench  $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused --threads 1,2,4,8 --reps 7
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256   # agreement + PPL matrix
```
(`turbohead-agreement`/`turbohead-ppl --npz <clusters.npz>` are lighter single-flash-head spot-checks.)

## Architecture

Three packages, split so the deploy path carries no offline-only deps:

- **`surgery/`** — offline, apply the method. `build_all.sh` is the per-model driver. `build_clusters.py`
  does balanced k-means (constrained Lloyd; exact `K·cap==V`, no padding; block-vectorized tail for
  heads whose rounds stall) → `Cnorm (D,K)`, `Wperm (K,cap,D)`, `Vmap (K,cap)`.
  `build_subgraph.py::make_flash_nodes()` is the reusable op-chain builder; `splice.py --backend
  {fused,onnx}` produces the spliced model. `quantize_head.py` builds the dense-head baselines.
- **`csrc/turbohead_op.cc`** — the `fused` backend's stage-2 custom CPU op `FlashHeadSelect` (emits the
  candidate shortlist; header-only against the ORT C/C++ API; `bash csrc/build.sh` → `libturbohead.so`).
- **`inference/decode_loop.py`** — the *only* file needed to run a spliced model. Self-contained: raw
  `onnxruntime` + numpy + tokenizer, no genai/torch. Doubles as the profiler. **Generic state handling**:
  discovers every `past*` input and seeds it from its declared shape, so it drives standard, hybrid
  (conv/SSM + sparse-index attention, e.g. LFM2.5), and embeds-in (Qwen3.5: numpy lookup from `head_W`
  + 3-D M-RoPE `position_ids`) models with no per-model config.
- **`eval/`** — gates: `benchmark.py` (`turbohead-bench`, speed matrix, median±std, **one model per
  subprocess** — two custom-op `.so` in one process segfault), `head_quality.py` (`turbohead-head-quality`,
  agreement + PPL on real hidden states via the extracted head subgraph), `agreement.py`/`ppl.py`
  (numpy-reference spot-checks). All hook the HF `lm_head` input over **detokenized** WikiText-2.
- **`experimental/`** — tried-idea POCs, one standalone folder each; core depends on none of it. Each may
  import core read-only but overrides locally when it diverges; promote a proven idea into core. See its README.

### The flash head op chain (logits-out, two stages)

1. **Stage 1 (coarse):** `sims = h · Cᵀ` over `K` centroids → `TopK(P)`. This `[K,D]` gemv at M=1 is the
   dominant head cost; it runs as int4 `MatMulNBits` (W4A16, `accuracy_level=4`) — ~9× faster than fp16
   at M=1 while holding 100% standalone argmax. **`MatMulNBits` is the only int gemv that wins at M=1**
   (it fuses dequant into the gemv); `MatMulInteger`/manual dequant were tried and lost.
2. **Stage 2 (refine):** `Gather` the `P·cap` candidate rows, dot with `h` for exact logits, scatter
   into a `-1e9`-filled `(1,V)` base. Stage 2 stays **fp32** — the CPU EP has no fp16 matmul kernel
   (it auto-casts), so storing fp32 skips an implicit fp16→fp32 cast of the gathered rows.

### Graph output shapes (auto-detected from graph outputs in `decode_loop.py`)

- **logits-out:** graph emits `logits` `(1,1,V)`; argmax/sampling in Python. The `onnx` backend.
- **shortlist-out:** the `fused` op emits `cand_logits`/`cand_ids` (~`P·cap` scored tokens); V never
  materialized. The `fused` backend — **fastest, current default**.
- **token-out:** graph emits the next-token id directly. Greedy-only; not currently produced.

Sampling (`temperature>0`) does a probed-softmax over **only** the scored candidates, skipping the
full-vocab softmax the dense head pays — so sampling speeds up more than greedy.

## Key facts to keep in mind

- **Speedup scales with head share** (narrow `D` + big `V` + few/light layers ⇒ bigger win): 1.45×
  (danube3) → 5.37× (Gemma3-270M) @1t vs an **fp32** head. Against an **int8** head (genai default) the
  Amdahl ceiling is only ~1.33× (head ≈30% of the step) — both framings are in RESULTS. More threads
  *shrink* the gain (head is memory-bound; cores also speed the dense baseline). Best single-threaded.
- **Coverage caps full-distribution PPL.** A true token outside the probed set gets ≈0 probability.
  Raise `-P` to lift coverage, at a speed cost. **Read PPL within-model only** (absolute PPL varies by
  tokenizer/base-model — verified `H·Wᵀ`==model logits, so it's real, not a bug; WikiText is
  detokenized before scoring). **Top-1 agreement is the cross-model-comparable quality metric.**
- **Hybrid/embeds-in models** bench end-to-end via the decode loop's generic state path. Two custom-op
  `.so` in one process segfault → `turbohead-bench` runs one model per subprocess.
- `cap` must divide `V` exactly (`V=151936=2⁷·1187`). Downstream steps read `cap`/`K` from the `.npz`;
  only `-P`/`--stage1`/`--block-size` shape inference.
- `onnx.checker` is deliberately skipped on save — it rejects genai's `com.microsoft` contrib ops; ORT
  load validates instead. The dense head node is removed but its weight initializer stays (tied embed).
- **Buffer-shared KV** (`decode_loop.py` `share_kv`, on by default) pre-allocates one max-length
  OrtValue per growing KV tensor and binds past-in≡present-out so GQA writes in place (seqlens from a
  MAX-width mask). Byte-identical, ~halves the per-step-vs-length slope → ~1.2× decode @400t on
  Qwen3-0.6B fused, growing with context. Only the *user*-allocated-buffer IOBinding pattern works;
  binding ORT-*allocated* output buffers back as inputs segfaults (arena recycle). Auto-disabled for
  hybrids with fixed conv/SSM state. See the `Decoder` docstring + `experimental/buffershare/buffershare_poc.py`.
- Code comments mark deliberate simplifications with `ponytail:`; non-obvious ORT op/precision choices
  point to `docs/ORT_QUIRKS.md` for the measured rationale.
