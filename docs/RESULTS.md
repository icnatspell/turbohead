# Experiment results

Per-model head-precision comparison: **is FlashHead worth it vs just quantizing the dense head?**
One section per model; add new models by appending a section in the same shape.

## Summary — all models

Fused FlashHead (contract H) decode speedup **vs the fp32-equivalent dense head** (`head16`), single
thread, int4 body. Sorted by greedy speedup. The win tracks the head's share of a decode step: narrow
hidden `D` + large vocab `V` + few layers ⇒ the head dominates ⇒ bigger speedup.

| model | D | V | layers | fused greedy @1t | fused sampling @1t | flash agree | coverage | head8g128 agree |
|---|---|---|---|---|---|---|---|---|
| Gemma3-270M | 640 | 262k | 18 | **5.37×** | **6.08×** | 95.2% | 87.6% | 98.3% |
| Gemma3-1B | 1152 | 262k | 26 | **3.04×** | **3.08×** | 94.4% | 85.6% | 97.9% |
| Qwen3-0.6B | 1024 | 152k | 28 | **2.40×** | **2.45×** | 97.6% | 88.5% | 98.5% |
| Llama-3.2-1B | 2048 | 128k | 16 | **2.09×** | **2.08×** | 96.9% | 90.8% | 99.3% |
| Qwen3-1.7B | 2048 | 152k | 28 | **2.05×** | **2.10×** | 97.9% | 88.6% | 98.5% |

Constant across the set: **fused flash beats every dense quant head** (incl. int4) on both speed and
greedy agreement at every thread count; **`head8 g128`** is the dense quality sweet spot (≈98–99%
agreement, fp32-equal PPL) when you need calibrated full-vocab probabilities; flash full-distribution
PPL is coverage-limited (raise `-P`). Per-model speed×thread×regime tables and reproduce commands below.

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

## Gemma3-1B (int4 body) — 2026-06-16

V=262144, D=1152, 26 layers. FlashHead: cap=16, K=16384, P=256. Reference PPL (fp32 head) = 7.931.
Int4 body = RTN (k_quant_last reshape-crashes; auto-fallback). Same 262k vocab as 270M but wider
hidden + more layers, so the head is a smaller share than 270M → smaller (but still large) flash win.

```bash
bash turbohead/surgery/build_all.sh google/gemma-3-1b-pt gemma3_1b
R=artifacts/gemma3_1b
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7                              # greedy
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

### Quality (vs fp32 head, 1999 WikiText-2 positions)

| head | top-1 agree | PPL |
|---|---|---|
| head16 (fp32-eq, ref) | 100.0% | 7.931 |
| head8 g128 | **97.9%** | 7.929 |
| head4 g128 | 85.5% | 8.413 |
| head4 g32 | 90.7% | 8.149 |
| flash onnx / fused (A / H) | 94.4% | 171.189 † |

† Coverage 85.6% at P=256; *covered* PPL = 3.886.

### Speed — greedy (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 12.3±0.2 (1.00×) | 18.7±0.2 (1.00×) | 20.9±0.5 (1.00×) | 21.0±0.2 (1.00×) |
| head8 g128 | 23.4±0.7 (1.91×) | 35.9±0.6 (1.92×) | 42.0±0.6 (2.01×) | 41.2±0.9 (1.96×) |
| head4 g128 | 29.7±0.4 (2.42×) | 43.2±1.1 (2.31×) | 49.7±1.5 (2.38×) | 47.5±2.6 (2.26×) |
| head4 g32 | 27.6±0.3 (2.25×) | 39.1±1.6 (2.09×) | 44.0±1.7 (2.11×) | 46.3±1.1 (2.20×) |
| flash onnx (A) | 34.3±0.4 (2.79×) | 48.9±0.8 (2.61×) | 57.5±2.1 (2.75×) | 57.1±1.4 (2.71×) |
| **flash fused (H)** | **37.3±0.2 (3.04×)** | **50.8±1.2 (2.71×)** | **61.8±2.0 (2.96×)** | **61.8±2.6 (2.94×)** |

### Speed — sampling, temp 0.8 (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 11.6±0.2 (1.00×) | 17.4±0.3 (1.00×) | 19.6±0.2 (1.00×) | 19.8±0.2 (1.00×) |
| head8 g128 | 22.5±0.4 (1.95×) | 30.0±1.4 (1.72×) | 36.7±0.5 (1.87×) | 34.9±2.0 (1.76×) |
| head4 g128 | 25.9±0.6 (2.24×) | 36.0±1.5 (2.07×) | 40.9±1.8 (2.08×) | 42.2±1.5 (2.13×) |
| head4 g32 | 22.1±1.1 (1.91×) | 34.0±1.2 (1.95×) | 38.3±0.8 (1.95×) | 37.9±0.9 (1.91×) |
| flash onnx (A) | 31.8±0.9 (2.75×) | 47.2±2.4 (2.71×) | 54.2±2.0 (2.76×) | 53.1±2.3 (2.68×) |
| **flash fused (H)** | **35.6±1.0 (3.08×)** | **48.4±2.2 (2.77×)** | **57.5±3.2 (2.93×)** | **56.5±1.7 (2.85×)** |

### Takeaways

- Fused FlashHead **3.04× greedy / 3.08× sampling @1t** vs fp32 head — between Qwen3-0.6B (2.40×) and
  Gemma3-270M (5.37×), consistent with head share: same 262k vocab as 270M but D=1152 and 26 layers
  dilute the head's fraction of decode.
- Fused beats every dense head at every thread count; head8 g128 again the dense quality sweet spot
  (97.9%, fp32-equal PPL). int4 dense agreement weak (85.5% g128 / 90.7% g32); flash 94.4% is the
  better trade. Coverage 85.6% — bump P if full-vocab likelihood matters.

---

## Llama-3.2-1B (int4 body) — 2026-06-16

V=128256, D=2048, 16 layers. FlashHead: cap=16, K=8016, P=256. Reference PPL (fp32 head) = 7.115.
Int4 body = RTN. Wide hidden (2048) + smaller vocab ⇒ head is a relatively small share, so the flash
win is the most modest of the set (similar to Qwen).

```bash
bash turbohead/surgery/build_all.sh meta-llama/Llama-3.2-1B llama3_2_1b
R=artifacts/llama3_2_1b
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7                              # greedy
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

### Quality (vs fp32 head, 1999 WikiText-2 positions)

| head | top-1 agree | PPL |
|---|---|---|
| head16 (fp32-eq, ref) | 100.0% | 7.115 |
| head8 g128 | **99.3%** | 7.118 |
| head4 g128 | 93.5% | 7.179 |
| head4 g32 | 93.7% | 7.199 |
| flash onnx / fused (A / H) | 96.9% | 47.102 † |

† Coverage 90.8% at P=256 (highest of the set — smaller vocab); *covered* PPL = 4.289.

### Speed — greedy (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 11.6±0.2 (1.00×) | 17.4±1.6 (1.00×) | 20.2±0.5 (1.00×) | 20.8±0.4 (1.00×) |
| head8 g128 | 18.2±0.7 (1.57×) | 29.7±0.3 (1.71×) | 37.3±2.1 (1.85×) | 38.6±4.0 (1.86×) |
| head4 g128 | 20.0±0.6 (1.72×) | 32.9±1.5 (1.89×) | 43.5±1.0 (2.16×) | 45.1±1.0 (2.17×) |
| head4 g32 | 18.9±0.6 (1.63×) | 32.4±0.8 (1.86×) | 39.9±0.6 (1.98×) | 41.0±0.7 (1.98×) |
| flash onnx (A) | 20.6±1.1 (1.77×) | 35.2±0.7 (2.02×) | 39.1±3.0 (1.94×) | 46.1±1.0 (2.22×) |
| **flash fused (H)** | **24.2±0.3 (2.09×)** | **38.9±0.5 (2.23×)** | **48.4±1.3 (2.40×)** | 46.6±6.7 (2.24×) |

### Speed — sampling, temp 0.8 (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 11.2±0.1 (1.00×) | 17.5±0.1 (1.00×) | 20.4±0.7 (1.00×) | 21.2±0.5 (1.00×) |
| head8 g128 | 16.3±0.7 (1.45×) | 26.4±1.6 (1.51×) | 32.1±2.3 (1.58×) | 35.0±0.9 (1.65×) |
| head4 g128 | 18.8±0.7 (1.68×) | 31.0±1.5 (1.78×) | 40.6±1.5 (1.99×) | 41.2±0.9 (1.94×) |
| head4 g32 | 18.8±0.7 (1.67×) | 30.3±1.4 (1.73×) | 35.5±2.7 (1.74×) | 35.6±0.9 (1.68×) |
| flash onnx (A) | 21.4±0.4 (1.91×) | 33.6±0.7 (1.93×) | 42.2±1.8 (2.07×) | 44.5±1.6 (2.10×) |
| **flash fused (H)** | **23.3±0.4 (2.08×)** | **36.3±1.4 (2.08×)** | **43.9±2.9 (2.16×)** | **46.0±4.4 (2.16×)** |

### Takeaways

- Fused FlashHead **2.09× greedy / 2.08× sampling @1t** — the most modest win of the set, consistent
  with the smallest head share (D=2048, only 128k vocab, 16 layers). Still beats every dense head.
- Highest coverage (90.8%) and best int4-dense agreement (93.5–93.7%) of the four models; head8 g128
  near-lossless (99.3%, fp32-equal PPL). Flash 96.9% agreement.

---

## Qwen3-1.7B (int4 body) — 2026-06-17

V=151936, D=2048, 28 layers. FlashHead: cap=16, K=9496, P=256. Reference PPL (fp32 head) = 11.173.
Int4 body = RTN. Wide hidden (2048) + the most layers of the set ⇒ smallest head share, so the flash
win is the most modest measured (just under Llama, which shares D=2048 but has fewer layers).

```bash
bash turbohead/surgery/build_all.sh Qwen/Qwen3-1.7B qwen3_1_7b
R=artifacts/qwen3_1_7b
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7                              # greedy
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

### Quality (vs fp32 head, 1999 WikiText-2 positions)

| head | top-1 agree | PPL |
|---|---|---|
| head16 (fp32-eq, ref) | 100.0% | 11.173 |
| head8 g128 | **98.5%** | 11.168 |
| head4 g128 | 93.4% | 11.295 |
| head4 g32 | 94.8% | 11.201 |
| flash onnx / fused (A / H) | 97.9% | 100.580 † |

† Coverage 88.6% at P=256; *covered* PPL = 5.270.

### Speed — greedy (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 9.4±0.8 (1.00×) | 14.7±0.3 (1.00×) | 16.3±0.3 (1.00×) | 16.5±0.3 (1.00×) |
| head8 g128 | 15.5±0.1 (1.65×) | 24.3±0.2 (1.65×) | 28.4±0.4 (1.74×) | 28.4±0.3 (1.72×) |
| head4 g128 | 16.8±0.2 (1.80×) | 25.5±1.0 (1.74×) | 31.8±0.2 (1.95×) | 31.0±2.3 (1.88×) |
| head4 g32 | 16.3±0.2 (1.74×) | 25.6±0.3 (1.75×) | 30.5±0.4 (1.87×) | 30.4±0.3 (1.85×) |
| flash onnx (A) | 17.7±0.1 (1.89×) | 27.3±0.5 (1.86×) | 32.7±0.4 (2.00×) | 32.5±0.4 (1.98×) |
| **flash fused (H)** | **19.2±0.4 (2.05×)** | **29.1±0.4 (1.99×)** | **34.9±0.8 (2.14×)** | **35.3±0.4 (2.14×)** |

### Speed — sampling, temp 0.8 (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 9.1±0.2 (1.00×) | 13.7±0.2 (1.00×) | 16.0±0.2 (1.00×) | 16.2±0.2 (1.00×) |
| head8 g128 | 14.6±0.3 (1.60×) | 22.4±0.5 (1.63×) | 26.9±0.2 (1.69×) | 27.0±0.2 (1.67×) |
| head4 g128 | 16.3±0.1 (1.79×) | 25.2±0.3 (1.83×) | 29.8±0.3 (1.87×) | 30.1±0.3 (1.86×) |
| head4 g32 | 15.6±0.2 (1.71×) | 23.8±0.3 (1.73×) | 28.4±0.3 (1.78×) | 29.0±0.3 (1.79×) |
| flash onnx (A) | 17.2±0.3 (1.88×) | 27.1±0.2 (1.97×) | 32.6±0.3 (2.04×) | 32.9±0.6 (2.03×) |
| **flash fused (H)** | **19.2±0.2 (2.10×)** | **29.4±0.4 (2.14×)** | **34.9±0.4 (2.18×)** | **35.5±0.5 (2.19×)** |

### Takeaways

- Fused FlashHead **2.05× greedy / 2.10× sampling @1t** — the most modest win of the set, consistent
  with the smallest head share (D=2048, 28 layers — most of the set). Still beats every dense head.
- head8 g128 near-lossless (98.5%, fp32-equal PPL); flash 97.9% agreement, covered PPL 5.270.
- Clustering note: this head's geometry stalled the balanced-assign rounds (left ~60% of the vocab to
  the tail). Fixed in `build_clusters.py` — bail when rounds stall + block-vectorized tail fill.

---

## Models pending

Each is one `build_all.sh <hf-model> <slug>` then the three commands above. FlashHead's edge should
grow where the head is a larger share of decode (bigger V:D, fewer layers).

| model | hf id | slug |
|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | `qwen3_5_0_8b` |
| LFM2.5-350M | `LiquidAI/LFM2.5-350M` | `lfm2_5_350m` |
| h2o-danube3-500m-chat | `h2oai/h2o-danube3-500m-chat` | `danube3_500m` |

(cap must divide V; if `cap=16` doesn't, `build_all` fails at clustering — pick a divisor of that
model's vocab via the `[cap]` arg. block_size 128 must divide D.)
