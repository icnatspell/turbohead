# TurboHead — a fast approximate LM head for quantized LLMs

For large vocabularies the final language-model head (the `hidden → vocab` projection) is one of
the most expensive ops in a decode step: a single dense `[V, D]` matmul over a vocab `V` that is
often 150k+. TurboHead replaces it with an **approximate clustering-based head** and splices that
directly into a quantized ONNX model, so the speedup is available to any `onnxruntime` deploy with
no custom runtime.

The method follows **FlashHead** ([paper](https://arxiv.org/abs/2603.14591),
[code](https://github.com/embedl/flash-head)):

1. **Cluster the vocab.** Balanced k-means over the head's weight rows groups the `V` tokens into
   `K` equal clusters of `cap = V/K` tokens each (the *cluster ratio*). Each cluster has a centroid.
2. **Stage 1 — coarse scoring.** Score the hidden state against the `K` centroids (`sims = h · Cᵀ`)
   and keep the top `P` clusters ("probes"). This is a small `[K, D]` matmul that tolerates low
   precision, so it runs in int4/int8 `MatMulNBits` (fused dequant gemv).
3. **Stage 2 — refine.** Gather only the `P·cap` candidate token rows and dot them with `h` to get
   exact logits for that shortlist; everything else is left unscored. Argmax (greedy) or a
   **probed-softmax** over just the candidates (sampling) picks the next token.

Net effect: instead of touching all `V` rows every step, you touch `K` centroids + `P·cap`
candidate rows. Quality is preserved as long as the true next token lands in the probed set
(*coverage*), which `P` controls.

This repo targets **CPU inference with ONNXRuntime** and ships the surgery, a self-contained decode
loop, and the quality/speed gates. Full design notes in `docs/PLAN.md`; findings in
`docs/NEXT_STEPS.md`.

---

## Install

```bash
uv sync          # everything: surgery, the decode loop, and the quality/speed gates
# or:  pip install turbohead
```

This installs the full toolkit (onnx/torch/datasets included) — you apply the method *and* run/
evaluate it from one install. The `dev` group adds ruff/pytest/pyrefly for contributors.

## Usage

### Offline — apply TurboHead to your model

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

- **Cluster count** = the cluster ratio: `--cap N` is tokens per cluster (FlashHead's
  `DEFAULT_CLUSTER_RATIO=16`), giving `K = V/cap` clusters; or set `K` directly with `--clusters`.
  `cap` must divide `V`. Downstream steps read `cap`/`K` from the `.npz` — only this step needs it.
- `-P` (probes) and **stage-1 precision** (`fp16` | `int8` | `int4`, default int4) + quant
  `--block-size` are chosen at splice time. int4 is fastest; fp16 is the exact reference.

### Online — the decode loop

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
Flags: `--threads N`, `--reps R` (median tok/s), `--max-new M`, `--prompt STR`, `--temperature T`
/ `--seed N`, `--profile`. `--temperature 0` is greedy argmax; `>0` samples via **probed-softmax**
over only the scored candidate set — skipping the full-vocab softmax the dense head pays, so the
speedup is *larger* when sampling than when greedy.

### Measuring quality

Two views, both vs the original dense head on WikiText-2:

```bash
uv run turbohead-agreement --npz artifacts/clusters.npz    # top-1 agreement vs dense argmax (sweeps P)
uv run turbohead-ppl       --npz artifacts/clusters.npz    # dense vs flash perplexity + coverage
```

- **Top-1 agreement** — how often TurboHead's argmax matches the dense head's. This is the
  deploy-relevant greedy metric.
- **Coverage** — fraction of true next-tokens that fall inside the probed candidate set. This is
  the recall ceiling; raise `-P` to improve it. Full-distribution PPL is bounded by coverage (a
  target outside the set gets ≈0 probability), so report the two together.

### Profiling

The decode loop doubles as an exhaustive per-op profiler (ORT `enable_profiling`):

```bash
uv run turbohead-decode artifacts/<model>_flash --reps 5 --profile
```

It prints prefill/decode `model_run` medians and a per-op-type breakdown (ms/step, % of node time,
call count) plus a best-effort head-path rollup — i.e. exactly where each decode step's time goes.

## Results — Qwen3-0.6B, CPU, ONNXRuntime

Reference config: `Qwen/Qwen3-0.6B` (`V=151936`, `D=1024`), int4 body / int8 dense head baseline,
balanced clustering `cap=16 → K=9496` (no padding), `P=256`, stage-1 int4 `MatMulNBits`. Single
socket, `onnxruntime` CPU EP.

### Speed

| | 1 thread | 4 threads |
|---|---|---|
| **Decode speedup vs int4 baseline** | **1.21×** | **1.19×** |

The advantage erodes as threads grow (the head is memory-bound; more cores help the baseline's dense
head too) and **grows under sampling** (~1.35×), where the dense baseline pays a full-vocab softmax
every step that TurboHead skips.

### Where the speedup comes from (per decode step, 1 thread, `--profile`)

The body and attention are unchanged; **TurboHead only touches the head**. The dense int8 head
(~7.9 ms) is replaced by the flash head (~3.3 ms):

| component | baseline | TurboHead |
|---|---|---|
| transformer body (`MatMulNBits`, int4) | ~15.1 ms | ~15.2 ms |
| attention (`GroupQueryAttention`) | 2.08 ms | 2.04 ms |
| **head — dense `[V,D]` matmul** | **7.87 ms** | — |
| **head — stage-1 centroid scoring (int4)** | — | ~0.16 ms |
| **head — stage-2 gather (`P·cap` rows)** | — | 1.49 ms |
| **head — stage-2 dot + topk/scatter** | — | ~1.65 ms |
| **decode step total** | **26.55 ms** | **22.63 ms** |

So the head drops from 7.9 ms → 3.3 ms. The remaining head cost is dominated by the stage-2 **Gather**
(1.49 ms) and dot (1.46 ms), not stage-1 — int4 already shrank stage-1 to ~0.16 ms. The hard ceiling
here is Amdahl: the head was only ~30% of the step, so even a free head caps the speedup at ~1.33×.

### Quality (vs dense head, WikiText-2)

| metric | value | meaning |
|---|---|---|
| **Top-1 agreement** (P=256) | **97.6%** | greedy next-token matches the dense head |
| Standalone subgraph argmax | **100%** | int4 stage-1 preserves argmax vs fp16 reference |
| Candidate coverage (P=256) | 89.4% | true token is inside the probed set; ↑ with P |
| Dense PPL / coverage-limited flash PPL | 10.9 / 82.6 | full-distribution PPL is bounded by coverage |

Greedy decoding — what most deploys use — is preserved at **97.6%** agreement with no measurable
quality loss on deterministic prompts. The full-distribution PPL gap is purely **coverage**: ~10% of
targets fall outside the P=256 candidate set and get ≈0 probability. Raising `P` trades speed for
coverage; `P=256` is the chosen CPU operating point.

> int4 stage-1 is the key precision trick: stage-1 centroid scoring (the dominant head gemv at M=1)
> runs as int4 `MatMulNBits` (W4A16, `accuracy_level=4`), 9× faster than fp16 on that gemv while
> preserving 100% standalone argmax. Earlier int4 attempts via `MatMulInteger`/manual dequant were
> *slower* — `MatMulNBits` fuses dequant into the gemv and is the only int gemv that wins at M=1.
