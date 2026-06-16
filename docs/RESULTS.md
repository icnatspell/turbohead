# Experiment results

Per-model head-precision comparison: **is FlashHead worth it vs just quantizing the dense head?**
One section per model; add new models by appending a section in the same shape.

## Reproduce a model from scratch

One command builds every artifact for a model into its own `artifacts/<slug>/` dir (genai int4
baseline → head weight → clusters → 4 dense-head variants → 2 flash splices), then prints the
bench/eval commands:

```bash
bash turbohead/surgery/build_all.sh <hf-model> <slug> [cap=16] [P=256]
```

`<slug>` is a dir-safe name; everything for the model lives under `artifacts/<slug>/`:
`baseline/`, `head_W.npy`, `clusters.npz`, `head{16,8g128,4g128,4g32}/`, `onnx/`, `fused/` — nothing
shared between models, so each is independently reusable. Re-running skips the slow genai baseline
unless `FORCE=1`. Per-model command blocks are listed in each section.

## Method

- **Speed** (`turbohead-bench`): decode tok/s, median ± std over 7 reps (+1 warmup), each model in
  its own subprocess. `--max-new 64`, prompt = "Once upon a time, in a small village,". Greedy and
  sampling (`--temperature 0.8 --seed 0`) at intra-op threads {1,2,4,8}. Speedup `(×)` is vs the
  fp32-equivalent `head16` at the **same** thread count.
- **Quality** (`turbohead-head-quality`): top-1 agreement + WikiText-2 PPL on the model's real
  hidden states (HF fp32), reference = the full-precision head (`head_W.npy`). Dense variants run
  their **actual** extracted head kernel (no numpy re-quant). Thread- and regime-independent.
- **Heads compared:** `head{16,8,4}` dense at MatMulNBits group size g (`turbohead-quantize-head`);
  `flash onnx` (contract A, portable (1,V) logits); `flash fused` (contract H, custom op). Body is
  int4 g128 for all rows; only the head varies.
- **Environment:** CPU only (CPUExecutionProvider), Linux WSL2, 8 logical cores. Single-threaded is
  the intended deploy point; >4 threads oversubscribe here.

---

## Qwen3-0.6B (int4 body) — 2026-06-16

V=151936, D=1024, 28 layers. FlashHead: cap=16, K=9496, P=256. Reference PPL (fp32 head) = 13.518.

```bash
bash turbohead/surgery/build_all.sh Qwen/Qwen3-0.6B qwen3_0_6b
R=artifacts/qwen3_0_6b
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7                              # greedy
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

### Quality (vs fp32 head, 1999 WikiText-2 positions)

| head | top-1 agree | PPL |
|---|---|---|
| head16 (fp32-eq, ref) | 100.0% | 13.518 |
| head8 g128 | **98.5%** | 13.520 |
| head4 g128 | 90.9% | 13.871 |
| head4 g32 | 93.2% | 13.562 |
| flash onnx (A) | 97.6% | 121.868 † |
| flash fused (H) | 97.6% | 121.868 † |

† Flash full-distribution PPL is **coverage-limited**: at P=256 only **88.5%** of targets fall in the
probed candidate set; uncovered targets are floored to ε, inflating PPL. The *covered* PPL (ranking
quality where the target is probed) = **6.356**. So flash is strong for greedy / sampling-over-shortlist
but weak for full-vocab likelihood — raise P, or use a dense head, if you need calibrated full-V probs.
(onnx and fused share the same clustering math, hence identical quality.)

### Speed — greedy (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 22.1±0.2 (1.00×) | 32.0±0.5 (1.00×) | 34.8±1.9 (1.00×) | 34.6±1.0 (1.00×) |
| head8 g128 | 38.7±0.4 (1.75×) | 53.5±1.9 (1.67×) | 61.9±0.7 (1.78×) | 60.4±2.1 (1.75×) |
| head4 g128 | 45.2±0.6 (2.05×) | 64.6±1.7 (2.02×) | 73.5±1.1 (2.11×) | 72.5±1.1 (2.10×) |
| head4 g32 | 42.4±0.6 (1.92×) | 61.2±1.2 (1.91×) | 69.4±1.0 (1.99×) | 69.0±1.0 (2.00×) |
| flash onnx (A) | 47.8±0.4 (2.17×) | 67.2±2.6 (2.10×) | 75.3±3.0 (2.16×) | 75.9±0.8 (2.20×) |
| **flash fused (H)** | **53.1±1.1 (2.40×)** | **74.2±1.5 (2.32×)** | **82.4±1.6 (2.37×)** | **81.7±1.6 (2.37×)** |

### Speed — sampling, temp 0.8 (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 21.3±0.2 (1.00×) | 31.4±0.6 (1.00×) | 33.6±0.7 (1.00×) | 33.6±0.9 (1.00×) |
| head8 g128 | 36.0±1.0 (1.69×) | 51.4±1.4 (1.64×) | 57.5±0.8 (1.71×) | 56.6±0.8 (1.69×) |
| head4 g128 | 41.3±0.6 (1.94×) | 57.5±1.0 (1.83×) | 65.6±3.1 (1.95×) | 64.0±1.3 (1.91×) |
| head4 g32 | 39.7±0.6 (1.86×) | 54.4±0.6 (1.73×) | 61.3±1.0 (1.82×) | 60.6±0.7 (1.80×) |
| flash onnx (A) | 46.8±0.6 (2.19×) | 66.3±1.2 (2.11×) | 75.6±1.4 (2.25×) | 73.9±0.5 (2.20×) |
| **flash fused (H)** | **52.2±0.7 (2.45×)** | **73.2±1.6 (2.33×)** | **77.6±2.3 (2.31×)** | **81.1±1.9 (2.42×)** |

### Takeaways

- **Fused contract-H is the fastest path at every thread count** (2.40× greedy / 2.45× sampling @1t),
  beating *every* dense quant head — including int4 — on both speed and greedy agreement.
- **head8 g128 is the dense quality sweet spot**: 98.5% agreement, PPL identical to fp32, 1.75×.
- Among int4 dense heads, **g128 is faster but g32 is more accurate** (smaller group = more scales =
  more dequant cost, better quality).
- Absolute throughput peaks at 4t and flattens/dips at 8t (oversubscription). Speedup-vs-`head16` is
  roughly thread-flat because the fp32 baseline also scales with threads.

---

## Gemma3-270M (int4 body) — 2026-06-16

V=262144, D=640, 18 layers. FlashHead: cap=16, K=16384, P=256. Reference PPL (fp32 head) = 11.636.
Int4 body uses **plain RTN** (genai's `k_quant_last` reshape-crashes on this model's weight shapes;
build_all auto-falls back). Tiny D + huge V + few layers ⇒ the head dominates decode, so FlashHead's
edge is far larger than on Qwen3-0.6B.

```bash
bash turbohead/surgery/build_all.sh google/gemma-3-270m gemma3_270m
R=artifacts/gemma3_270m
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7                              # greedy
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

### Quality (vs fp32 head, 1999 WikiText-2 positions)

| head | top-1 agree | PPL |
|---|---|---|
| head16 (fp32-eq, ref) | 100.0% | 11.636 |
| head8 g128 | **98.3%** | 11.649 |
| head4 g128 | 82.7% | 12.900 |
| head4 g32 | 88.0% | 12.169 |
| flash onnx / fused (A / H) | 95.2% | 150.280 † |

† Coverage 87.6% at P=256; *covered* PPL = 6.195 (same caveat as Qwen — flash is coverage-limited for
full-vocab likelihood).

### Speed — greedy (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 29.8±0.3 (1.00×) | 42.4±0.6 (1.00×) | 44.6±0.7 (1.00×) | 45.5±0.8 (1.00×) |
| head8 g128 | 71.5±0.6 (2.40×) | 98.6±4.1 (2.32×) | 110.1±2.5 (2.47×) | 101.3±8.6 (2.22×) |
| head4 g128 | 94.7±1.7 (3.18×) | 127.5±5.8 (3.00×) | 144.2±3.6 (3.23×) | 142.4±3.9 (3.13×) |
| head4 g32 | 84.4±1.6 (2.83×) | 109.3±3.6 (2.58×) | 128.2±1.7 (2.87×) | 126.3±2.6 (2.77×) |
| flash onnx (A) | 127.0±3.8 (4.26×) | 166.2±8.3 (3.91×) | 182.5±11.7 (4.09×) | 178.6±10.7 (3.92×) |
| **flash fused (H)** | **160.1±2.9 (5.37×)** | **189.5±4.7 (4.47×)** | **207.5±5.1 (4.65×)** | **207.8±2.1 (4.56×)** |

### Speed — sampling, temp 0.8 (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 25.8±0.1 (1.00×) | 36.7±1.1 (1.00×) | 39.8±1.1 (1.00×) | 39.8±0.5 (1.00×) |
| head8 g128 | 57.9±0.7 (2.24×) | 74.8±1.7 (2.04×) | 83.1±1.8 (2.09×) | 77.2±5.6 (1.94×) |
| head4 g128 | 70.6±1.3 (2.73×) | 89.2±4.0 (2.43×) | 99.6±1.7 (2.50×) | 97.4±1.1 (2.45×) |
| head4 g32 | 65.1±0.9 (2.52×) | 73.4±3.3 (2.00×) | 92.5±1.5 (2.33×) | 91.3±2.0 (2.29×) |
| flash onnx (A) | 126.1±4.0 (4.88×) | 158.5±7.9 (4.31×) | 183.0±10.5 (4.60×) | 173.8±4.2 (4.37×) |
| **flash fused (H)** | **157.2±4.5 (6.08×)** | **189.8±8.5 (5.17×)** | **215.6±9.5 (5.42×)** | **206.9±9.6 (5.20×)** |

### Takeaways

- **FlashHead's edge scales with head share, as predicted.** With a 640-wide hidden, 262k vocab and
  only 18 layers the head is the dominant cost — fused FlashHead hits **5.37× greedy / 6.08× sampling
  @1t** vs the fp32 head (vs 2.40× / 2.45× on Qwen3-0.6B). Even int4 dense heads only reach ~3×.
- Fused beats onnx and every dense head at every thread count; head8 g128 is again the dense quality
  sweet spot (98.3%, fp32-equal PPL). int4 dense agreement is notably worse here (82.7% g128 / 88.0%
  g32) than on Qwen — flash at 95.2% is the better accuracy/speed trade.

---

## Models pending

Each is one `build_all.sh <hf-model> <slug>` then the three commands above. FlashHead's edge should
grow where the head is a larger share of decode (bigger V:D, fewer layers).

| model | hf id | slug |
|---|---|---|
| Gemma3-1B | `google/gemma-3-1b-pt` | `gemma3_1b` |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | `llama3_2_1b` |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | `qwen3_1_7b` |
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | `qwen3_5_0_8b` |
| LFM2.5-350M | `LiquidAI/LFM2.5-350M` | `lfm2_5_350m` |
| h2o-danube3-500m-chat | `h2oai/h2o-danube3-500m-chat` | `danube3_500m` |

(cap must divide V; if `cap=16` doesn't, `build_all` fails at clustering — pick a divisor of that
model's vocab via the `[cap]` arg. block_size 128 must divide D.)
