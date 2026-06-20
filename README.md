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
decode loop to run it, and gates to measure quality and speed.

### Where things live

| path | what |
|---|---|
| `turbohead/surgery/` | **offline** — apply the method. `build_all.sh` (one-command pipeline), `build_clusters.py` (balanced k-means), `build_subgraph.py` (the op-chain), `splice.py`, `quantize_head.py` (dense baselines) |
| `turbohead/inference/decode_loop.py` | **deploy** — the only file needed to *run* a spliced model (raw ORT + numpy + tokenizer; also the profiler). Drives standard, hybrid, and embeds-in models |
| `turbohead/eval/` | **gates** — `benchmark.py` (speed matrix), `head_quality.py` (agreement/PPL), `agreement.py`/`ppl.py` (spot-checks) |
| `csrc/turbohead_op.cc` | the fused stage-2 custom CPU op (`build.sh` → `libturbohead.so`) |
| `docs/RESULTS.md` | **the numbers** — 8-model speed×quality matrix, key findings, per-model reproduce commands, open items |
| `docs/ORT_QUIRKS.md` | measured ONNXRuntime CPU operator quirks (the *why* behind the precision/op choices) |
| `experimental/` | tried-idea POCs, one standalone folder each (the evidence behind what shipped vs got parked); core depends on none of it |

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

A one-time transform that turns a Hugging Face model into spliced ONNX dirs. **One command** builds
the whole artifact tree for a model — int4 baseline → head weight → clusters → dense-head variants →
two flash splices (portable `onnx`, fused custom-op `fused`) — into its own reusable `artifacts/<slug>/`:

```bash
bash turbohead/surgery/build_all.sh <hf-model> <slug>      # e.g. Qwen/Qwen3-0.6B qwen3_0_6b
# -> artifacts/<slug>/{baseline, head_W.npy, clusters.npz, head{16,8g128,4g128,4g32}, onnx, fused, logs}
```

It's idempotent (re-running reuses the slow genai baseline + clustering unless `FORCE=1`), tees the
full build log to `artifacts/<slug>/logs/`, and prints the bench/eval commands to run next. The int4
body defaults to plain RTN (robust across architectures). Then run / measure with the `onnx` or
`fused` dir — see [docs/RESULTS.md](docs/RESULTS.md) for the per-model command blocks.

<details><summary>Or step by step (what <code>build_all.sh</code> runs, here for one model)</summary>

```bash
R=artifacts/qwen3_0_6b
# 1. int4 baseline ONNX (genai model builder)
MODEL=Qwen/Qwen3-0.6B OUT=$R/baseline bash turbohead/surgery/convert_baseline.sh
# 2. dump the fp32 head weight (= tied embedding)
uv run turbohead-extract-head   --model Qwen/Qwen3-0.6B --out $R/head_W.npy
# 3. balanced k-means clustering assets        (cap=16; or --cap 32 / --clusters K)
uv run turbohead-build-clusters --head $R/head_W.npy --out $R/clusters.npz --cap 16
# 4. splice TurboHead in: portable (logits-out) and fused (shortlist-out, fastest)
uv run turbohead-splice --backend onnx  --src $R/baseline --npz $R/clusters.npz --head $R/head_W.npy -P 256 --dst $R/onnx
uv run turbohead-splice --backend fused --src $R/baseline --npz $R/clusters.npz --head $R/head_W.npy -P 256 --dst $R/fused
# (build_all also builds dense-head baselines via turbohead-quantize-head --bits {16,8,4} for comparison)
```
</details>

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
uv run turbohead-decode artifacts/<slug>/fused --reps 5      # or .../onnx for the portable backend
```

```python
from turbohead.inference.decode_loop import Decoder
dec = Decoder("artifacts/<slug>/fused", threads=1)    # dims/output-shape/tokenizer auto-detected
ids = dec.tok("Once upon a time,")["input_ids"]
out, tok_s = dec.generate(ids, max_new=64)            # temperature=0.0 greedy; >0 samples
print(dec.tok.decode(out))
```

The loop auto-detects the graph output shape (**logits-out** = full `(1,V)` logits, the `onnx`
backend; **shortlist-out** = the fused op's candidate shortlist, the `fused` backend; **token-out** =
the id directly) and the KV/SSM state layout — so it drives standard, hybrid (conv/SSM + attention),
and embeddings-in models with no per-model config.

| flag | default | what it does |
|---|---|---|
| `--threads N` | `1` | intra-op threads |
| `--max-new M` | `64` | tokens to generate |
| `--temperature T` | `0` | `0` runs greedy argmax; `>0` samples with probed-softmax over the candidate set, skipping the full-vocab softmax the dense head pays, so sampling speeds up more than greedy |
| `--seed N` | none | RNG seed for sampling |
| `--reps R` | `1` | benchmark: report median tok/s over `R` runs |
| `--prompt STR` | demo text | input prompt |
| `--profile` | off | dump the per-op breakdown (see below) |

### Measuring speed and quality

Two reusable harnesses, both run per-model against the dense head on WikiText-2:

```bash
R=artifacts/qwen3_0_6b
# speed matrix: decode tok/s, median±std over reps, each head in its own subprocess
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused --threads 1,2,4,8 --reps 7
uv run turbohead-bench $R/... --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling regime
# quality matrix: top-1 agreement + WikiText PPL vs the fp32 head, per head variant
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

`build_all.sh` prints these three lines pre-filled for the model you built. Reading the numbers:

- **Top-1 agreement** — how often TurboHead's argmax matches the dense head's. Greedy deploys care
  about this; it's the **cross-model-comparable** quality metric.
- **Coverage** — fraction of true next-tokens inside the probed candidate set (the recall ceiling).
  Raise `-P` to lift it. Coverage caps full-distribution PPL: an uncovered target gets ≈0 probability,
  so read the two together — and read **PPL only within a model** (absolute PPL varies by tokenizer/
  base-model; WikiText is detokenized before scoring).

(`turbohead-agreement --npz <clusters.npz>` and `turbohead-ppl --npz <clusters.npz>` are lighter
single-model spot-checks of just the flash head.)

### Profiling

The decode loop doubles as a per-op profiler (ORT `enable_profiling`):

```bash
uv run turbohead-decode artifacts/<slug>/fused --reps 5 --profile
```

It prints prefill and decode `model_run` medians, a per-op-type breakdown (ms/step, % of node time,
call count), and a head-path rollup, so you see where each decode step spends its time.

## Backends: portable ONNX vs fused custom op

`turbohead-splice` emits one of two head implementations. Both share the same clustering math and
produce identical quality; the decode loop auto-detects which one a model uses from its graph outputs.

- **`--backend onnx` (logits-out)** — pure ONNX standard ops (`MatMulNBits` → `TopK` → `Gather` →
  matmul → `ScatterElements`), producing full `(1, V)` logits. Runs on any stock `onnxruntime`, no
  native code. The portable path.
- **`--backend fused` (shortlist-out)** — same stage 1, but stage 2 collapses into a single custom CPU
  op, `turbohead::FlashHeadSelect` (`csrc/turbohead_op.cc`). The fastest path.

### The fused kernel

The spliced stage-2 op-chain (Gather the `P·cap` candidate rows → MatMul with `h` → Concat the
always-scored specials → Scatter into a `(1, V)` buffer) is several ops that each materialize an
intermediate — the costly one being the `(P·cap, D)` gather. The custom op collapses all of it into
one pass: for each probed cluster it dots that cluster's `cap` weight rows with `h` straight out of
`Wperm`, reading each candidate row exactly once with no `(P·cap, D)` materialization, and emits just
the **candidate shortlist** — `cand_logits` + `cand_ids` for the ~`P·cap` scored tokens. Python then
takes an argmax (greedy) or a softmax over the shortlist (sampling), skipping the full `(1, V)` logits,
the scatter, and the full-vocab softmax the dense head pays — which is why sampling speeds up even more
than greedy.

The kernel is header-only against the ORT C/C++ custom-op API (no `libonnxruntime` link), built by
`bash csrc/build.sh` → `csrc/libturbohead.so`; `build_all.sh` compiles it once and ships a copy in each
`fused/` dir, and the decode loop registers it via `SessionOptions.register_custom_ops_library`. The
dot-product loop is deliberately **single-threaded**: it streams ~`P·cap·D·4` ≈ 17 MB of weight rows
per step, so it's memory-bandwidth-bound and one thread already saturates the bandwidth that matters
(`-ffast-math` lets the reduction auto-vectorize; OpenMP over the clusters tested *slower* — fork/join
+ oversubscription against ORT's idle threads). The body's heavy matmuls still use all ORT threads;
this ~2 ms head isn't where the cores are needed.

## Results

Across eight models on the CPU EP (incl. hybrid conv/SSM + attention models like LFM2.5 and
Qwen3.5-0.8B), fused TurboHead decodes **1.45×–5.37× faster than an fp32-equivalent dense head** at one
thread, while matching its greedy next-token choice 94–98% of the time. The win scales with the head's
share of a decode step — narrow hidden `D`, large vocab `V`, few layers ⇒ the head dominates ⇒ bigger
speedup:

| model | D / V / layers | fused greedy @1t | fused sampling @1t | flash top-1 agree |
|---|---|---|---|---|
| Gemma3-270M | 640 / 262k / 18 | **5.37×** | **6.08×** | 94.8% |
| Qwen3-0.6B | 1024 / 152k / 28 | **2.40×** | **2.45×** | 96.8% |
| Qwen3-1.7B | 2048 / 152k / 28 | **2.05×** | **2.10×** | 97.2% |

Speedup is vs `head16` (an fp32 dense head on the same int4 body) at the same thread count; fused flash
also beats every *quantized* dense head (incl. int4) on both speed and agreement. Full tables — all
eight models, 1/2/4/8 threads, greedy + sampling, agreement + PPL + coverage — in
**[docs/RESULTS.md](docs/RESULTS.md)**. More threads shrink the speedup (the head is memory-bound and
extra cores also speed the dense baseline), so single-threaded is the intended deploy point.

### Anatomy of a decode step (Qwen3-0.6B, 1 thread, `--profile`)

TurboHead changes only the head; body and attention stay the same. Against genai's default int8 dense
head, it swaps a ~7.9 ms head for a ~3.3 ms one:

| component | dense head | TurboHead |
|---|---|---|
| transformer body (`MatMulNBits`, int4) | ~15.1 ms | ~15.2 ms |
| attention (`GroupQueryAttention`) | 2.08 ms | 2.04 ms |
| **head, dense `[V,D]` matmul** | **7.87 ms** | — |
| **head, stage-1 centroid scoring (int4)** | — | ~0.16 ms |
| **head, stage-2 gather (`P·cap` rows)** | — | 1.49 ms |
| **head, stage-2 dot + topk/scatter** | — | ~1.65 ms |
| **decode step total** | **26.55 ms** | **22.63 ms** |

The stage-2 Gather (1.49 ms) and dot (1.46 ms) dominate what remains in the standard-op path; int4
already shrank stage-1 to ~0.16 ms, and the fused kernel folds the gather away entirely. Amdahl sets
the ceiling on *this* baseline: with the head at ~30% of the step, even a free head caps the speedup at
~1.33× vs the int8 head — the larger headline numbers above are measured against an fp32 head, which is
itself slower.

### Quality (Qwen3-0.6B, vs dense head, WikiText-2)

| metric | value | meaning |
|---|---|---|
| **Top-1 agreement** (P=256) | **96.8%** | greedy next-token matches the dense head |
| Standalone subgraph argmax | **100%** | int4 stage-1 preserves argmax vs the fp16 reference |
| Candidate coverage (P=256) | 87.9% | true token sits inside the probed set; rises with P |
| Dense / covered / full-dist flash PPL | 13.0 / 6.4 / 144.8 | full-distribution PPL tracks coverage |

Greedy decoding — what most deploys use — holds 96.8% agreement with no measurable quality loss on
deterministic prompts. Coverage explains the full-distribution PPL gap: ~12% of targets fall outside
the P=256 candidate set and get ≈0 probability (the *covered* PPL, where the target is in the set, is
6.4). Raising `P` trades speed for coverage; we run `P=256` on CPU.

> int4 stage-1 carries the precision win. Stage-1 centroid scoring (the dominant head gemv at M=1)
> runs as int4 `MatMulNBits` (W4A16, `accuracy_level=4`), ~9× faster than fp16 on that gemv while
> holding 100% standalone argmax. Earlier int4 attempts through `MatMulInteger` and manual dequant
> ran slower; `MatMulNBits` fuses dequant into the gemv and is the only int gemv that wins at M=1.
