# TurboHead: a fast approximate LM head for quantized LLMs

A decode step spends close to a third of its time in the language-model head (30% on Qwen3-0.6B):
one dense `[V, D]` matmul that projects the hidden state onto a vocabulary `V` of 150k+ tokens.
TurboHead replaces that matmul with an approximate clustering-based head and splices it into a
quantized ONNX model, so any `onnxruntime` deploy gets the speedup without a custom runtime.

The method follows **FlashHead** ([paper](https://arxiv.org/abs/2603.14591),
[code](https://github.com/embedl/flash-head)):

1. **Cluster the vocab.** Balanced k-means over the head's weight rows groups the `V` tokens into
   `K` equal clusters of `cap = V/K` tokens each (the *cluster ratio*). Each cluster keeps a centroid.
2. **Stage 1, coarse scoring.** Score the hidden state against the `K` centroids (`sims = h · Cᵀ`)
   and keep the top `P` clusters ("probes"). This small `[K, D]` matmul tolerates low precision, so
   it runs in int4/int8 `MatMulNBits` (fused dequant gemv).
3. **Stage 2, refine.** Gather only the `P·cap` candidate token rows and dot them with `h` for exact
   logits on that shortlist, leaving everything else unscored. Argmax picks the greedy token; a
   probed-softmax over the candidates picks a sampled one.

Each step touches `K` centroids plus `P·cap` candidate rows instead of all `V`. Quality holds as
long as the true next token lands in the probed set; `P` controls that coverage.

This repo targets CPU inference with ONNXRuntime: surgery to apply the method, a self-contained
decode loop to run it, and gates to measure quality and speed. Design notes live in `docs/PLAN.md`,
findings in `docs/NEXT_STEPS.md`.

---

## Install

```bash
uv sync
# or:  pip install turbohead
```

One install gives you the full toolkit (onnx, torch, datasets): apply the method, run it, and
measure it. The `dev` group adds ruff, pytest, and pyrefly for contributors.

## Usage

### Offline: apply TurboHead to your model

A one-time transform that turns a quantized base model into a spliced `…_flash/` ONNX dir.

```bash
# 1. build the int4/int8 baseline ONNX model   -> artifacts/<model>_int4_cpu/
bash turbohead/surgery/convert_baseline.sh

# 2. dump the bf16 head weight                 -> artifacts/head_W.npy
uv run turbohead-extract-head

# 3. balanced k-means clustering assets        -> artifacts/clusters.npz
uv run turbohead-build-clusters                # default cap=16; or --cap 32 / --clusters 4748

# 4. splice TurboHead into the model           -> artifacts/<model>_flash/
uv run turbohead-splice -P 256 --stage1 int4 --block-size 128
```

`turbohead-build-clusters` sets the clustering. Pick `--cap` or `--clusters`, not both:

| flag | default | what it does |
|---|---|---|
| `--cap N` | `16` | tokens per cluster (FlashHead's cluster ratio); gives `K = V/cap` clusters. `cap` must divide `V`. Lower `cap` means more, smaller clusters |
| `--clusters K` | — | set the cluster count `K` directly instead of `cap` |

`turbohead-splice` sets the runtime tradeoff. Downstream steps read `cap`/`K` from the `.npz`, so
only these two flags shape inference:

| flag | default | what it does |
|---|---|---|
| `-P N` | `256` | probes: how many top clusters stage 2 refines. Higher `P` lifts coverage and accuracy, costs speed |
| `--stage1 {fp16,int8,int4}` | `int4` | centroid-scoring precision. `int4` is fastest; `fp16` reproduces the dense head exactly |
| `--block-size N` | `128` | `MatMulNBits` quant group size for int4/int8 stage 1 (ignored for fp16) |

### Online: the decode loop

The deploy path: a raw-`onnxruntime` greedy/sampling loop with a manual KV cache. No genai, no torch.

```bash
uv run turbohead-decode artifacts/<model>_flash --reps 5
```

```python
from turbohead.inference.decode_loop import Decoder
dec = Decoder("artifacts/<model>_flash", threads=1)   # dims/contract/tokenizer auto-detected
ids = dec.tok("Once upon a time,")["input_ids"]
out, tok_s = dec.generate(ids, max_new=64)            # temperature=0.0 greedy; >0 samples
print(dec.tok.decode(out))
```

The loop auto-detects the head contract (A = logits-out, B = token-out) and manages the KV cache.

| flag | default | what it does |
|---|---|---|
| `--threads N` | `1` | intra-op threads |
| `--max-new M` | `64` | tokens to generate |
| `--temperature T` | `0` | `0` runs greedy argmax; `>0` samples with probed-softmax over the candidate set, skipping the full-vocab softmax the dense head pays, so sampling speeds up more than greedy |
| `--seed N` | none | RNG seed for sampling |
| `--reps R` | `1` | benchmark: report median tok/s over `R` runs |
| `--prompt STR` | demo text | input prompt |
| `--profile` | off | dump the per-op breakdown (see below) |

### Measuring quality

Two views, both against the original dense head on WikiText-2:

```bash
uv run turbohead-agreement --npz artifacts/clusters.npz    # top-1 agreement vs dense argmax (sweeps P)
uv run turbohead-ppl       --npz artifacts/clusters.npz    # dense vs flash perplexity + coverage
```

- **Top-1 agreement** counts how often TurboHead's argmax matches the dense head's. Greedy deploys
  care about this number.
- **Coverage** is the fraction of true next-tokens that land inside the probed candidate set, the
  recall ceiling. Raise `-P` to lift it. Coverage caps full-distribution PPL: a target outside the
  set gets ≈0 probability, so read the two together.

### Profiling

The decode loop doubles as a per-op profiler (ORT `enable_profiling`):

```bash
uv run turbohead-decode artifacts/<model>_flash --reps 5 --profile
```

It prints prefill and decode `model_run` medians, a per-op-type breakdown (ms/step, % of node time,
call count), and a head-path rollup, so you see where each decode step spends its time.

## Results: Qwen3-0.6B, CPU, ONNXRuntime

Reference config: `Qwen/Qwen3-0.6B` (`V=151936`, `D=1024`), int4 body / int8 dense head baseline,
balanced clustering `cap=16 → K=9496` (no padding), `P=256`, stage-1 int4 `MatMulNBits`. Single
socket, `onnxruntime` CPU EP.

### Speed

| | 1 thread | 4 threads |
|---|---|---|
| **Decode speedup vs int4 baseline** | **1.21×** | **1.19×** |

More threads shrink the gap: the head is memory-bound, and extra cores speed up the baseline's dense
head too. Sampling widens it to ~1.35×, where the dense baseline pays a full-vocab softmax every step
that TurboHead skips.

### Where the speedup comes from (per decode step, 1 thread, `--profile`)

TurboHead changes only the head; body and attention stay the same. It swaps the dense int8 head
(~7.9 ms) for the flash head (~3.3 ms):

| component | baseline | TurboHead |
|---|---|---|
| transformer body (`MatMulNBits`, int4) | ~15.1 ms | ~15.2 ms |
| attention (`GroupQueryAttention`) | 2.08 ms | 2.04 ms |
| **head, dense `[V,D]` matmul** | **7.87 ms** | — |
| **head, stage-1 centroid scoring (int4)** | — | ~0.16 ms |
| **head, stage-2 gather (`P·cap` rows)** | — | 1.49 ms |
| **head, stage-2 dot + topk/scatter** | — | ~1.65 ms |
| **decode step total** | **26.55 ms** | **22.63 ms** |

The head drops from 7.9 ms to 3.3 ms. The stage-2 Gather (1.49 ms) and dot (1.46 ms) dominate what
remains; int4 already shrank stage-1 to ~0.16 ms. Amdahl sets the ceiling: the head was ~30% of the
step, so even a free head caps the speedup at ~1.33×.

### Quality (vs dense head, WikiText-2)

| metric | value | meaning |
|---|---|---|
| **Top-1 agreement** (P=256) | **97.6%** | greedy next-token matches the dense head |
| Standalone subgraph argmax | **100%** | int4 stage-1 preserves argmax vs the fp16 reference |
| Candidate coverage (P=256) | 89.4% | true token sits inside the probed set; rises with P |
| Dense PPL / coverage-limited flash PPL | 10.9 / 82.6 | full-distribution PPL tracks coverage |

Greedy decoding, what most deploys use, holds 97.6% agreement with no measurable quality loss on
deterministic prompts. Coverage explains the full-distribution PPL gap: ~10% of targets fall outside
the P=256 candidate set and get ≈0 probability. Raising `P` trades speed for coverage; we run `P=256`
on CPU.

> int4 stage-1 carries the precision win. Stage-1 centroid scoring (the dominant head gemv at M=1)
> runs as int4 `MatMulNBits` (W4A16, `accuracy_level=4`), 9× faster than fp16 on that gemv while
> holding 100% standalone argmax. Earlier int4 attempts through `MatMulInteger` and manual dequant
> ran slower; `MatMulNBits` fuses dequant into the gemv and is the only int gemv that wins at M=1.
