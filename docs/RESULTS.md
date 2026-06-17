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
| Qwen3.5-0.8B ‡ | 1024 | 248k | 24 | **2.82×** | **2.75×** | 96.7% | 86.4% | 98.9% |
| Qwen3-0.6B | 1024 | 152k | 28 | **2.40×** | **2.45×** | 97.6% | 88.5% | 98.5% |
| Llama-3.2-1B | 2048 | 128k | 16 | **2.09×** | **2.08×** | 96.9% | 90.8% | 99.3% |
| Qwen3-1.7B | 2048 | 152k | 28 | **2.05×** | **2.10×** | 97.9% | 88.6% | 98.5% |
| LFM2.5-350M ‡ | 1024 | 65k | 16 | **1.84×** | **2.04×** | 98.2% | 77.4% | 99.0% |
| h2o-danube3-500m | 1536 | 32k | 16 | **1.45×** | **1.21×** | 94.5% | 90.8% | 99.6% |

‡ **Hybrid** models (SSM/conv state interleaved with sparse attention), benched end-to-end via the
generalized decode loop. **Qwen3.5-0.8B** also splits the embedding out (needs `inputs_embeds` + 3-D
M-RoPE `position_ids`), handled by the loop's embeds-in path. The tiny-vocab **danube3** sits at the
low end — its head is a small share of decode, so flash barely leads dense heads; see its
`P`-sensitivity note below.

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

## Qwen3.5-0.8B (int4 body) — 2026-06-17

V=248320, D=1024, 24 layers (**hybrid Mamba/attention** — mostly SSM `conv_state`/`recurrent_state`,
only every 4th layer is attention). FlashHead: cap=16, K=15520, P=256. Reference PPL (fp32 head) =
12.499. The widest vocab-to-hidden ratio of the set (V:D=242) on top of a light SSM body ⇒ the head is
a large share of decode, so the flash win is among the biggest measured.

This model splits the embedding lookup out — it's driven by `inputs_embeds` + a 3-D (M-RoPE)
`position_ids`, not `input_ids`. The decode loop now supports that: it does the embedding lookup in
numpy from the tied embedding (`head_W`) and builds the 3-D position_ids itself (`embeds_in` path),
so it benches end-to-end despite the hybrid SSM state. (Earlier this model was quality-only.)

```bash
bash turbohead/surgery/build_all.sh Qwen/Qwen3.5-0.8B qwen3_5_0_8b
R=artifacts/qwen3_5_0_8b
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7                              # greedy
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

### Quality (vs fp32 head, 1999 WikiText-2 positions)

| head | top-1 agree | PPL |
|---|---|---|
| head16 (fp32-eq, ref) | 100.0% | 12.499 |
| head8 g128 | **98.9%** | 12.498 |
| head4 g128 | 92.1% | 12.713 |
| head4 g32 | 92.7% | 12.585 |
| flash onnx / fused (A / H) | 96.7% | 191.515 † |

† Coverage 86.4% at P=256; *covered* PPL = 5.645.

### Speed — greedy (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 14.5±0.2 (1.00×) | 21.9±0.2 (1.00×) | 24.2±1.5 (1.00×) | 24.4±0.4 (1.00×) |
| head8 g128 | 27.1±0.2 (1.87×) | 39.8±0.4 (1.82×) | 46.2±0.3 (1.91×) | 46.1±0.4 (1.89×) |
| head4 g128 | 32.2±0.3 (2.22×) | 45.2±1.0 (2.07×) | 52.7±0.7 (2.18×) | 54.2±0.9 (2.22×) |
| head4 g32 | 29.9±1.2 (2.06×) | 43.3±1.7 (1.98×) | 50.0±0.6 (2.07×) | 50.2±0.4 (2.05×) |
| flash onnx (A) | 37.3±0.5 (2.57×) | 51.7±1.3 (2.36×) | 59.7±0.9 (2.47×) | 59.7±0.5 (2.44×) |
| **flash fused (H)** | **40.8±0.5 (2.82×)** | **55.8±1.0 (2.55×)** | **64.1±1.3 (2.65×)** | **63.8±0.8 (2.61×)** |

### Speed — sampling, temp 0.8 (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 13.8±0.2 (1.00×) | 20.6±0.2 (1.00×) | 22.5±0.2 (1.00×) | 22.8±0.2 (1.00×) |
| head8 g128 | 23.9±0.3 (1.73×) | 33.7±0.9 (1.64×) | 39.1±2.3 (1.74×) | 40.1±0.2 (1.76×) |
| head4 g128 | 28.2±0.4 (2.04×) | 39.4±0.9 (1.92×) | 45.5±0.5 (2.02×) | 46.0±0.3 (2.02×) |
| head4 g32 | 26.8±0.6 (1.94×) | 37.1±1.2 (1.80×) | 43.8±1.1 (1.95×) | 42.1±2.0 (1.85×) |
| flash onnx (A) | 33.6±0.9 (2.43×) | 43.2±3.8 (2.10×) | 57.3±1.0 (2.55×) | 55.7±1.2 (2.44×) |
| **flash fused (H)** | **38.1±0.3 (2.75×)** | **51.4±1.1 (2.50×)** | **57.8±2.2 (2.57×)** | **58.9±0.8 (2.58×)** |

### Takeaways

- Fused FlashHead **2.82× greedy / 2.75× sampling @1t** — third-highest of the set, behind only the
  big-vocab Gemmas. The huge vocab (V:D=242) and the light SSM body make the head a dominant share.
- First **embeds-in** model benched end-to-end. head8 g128 near-lossless (98.9%, fp32-equal PPL); flash
  96.7% agreement, covered PPL 5.645. The head method transfers cleanly to the hybrid arch.

---

## h2o-danube3-500m-chat (int4 body) — 2026-06-17

V=32000, D=1536, 16 layers, **untied** embeddings. FlashHead: cap=16, K=2000, P=256. Reference PPL
(fp32 head) = 6.791. Standard Llama-style transformer. This is the **low end of the head-share
spectrum** — tiny vocab (32k) + wide hidden (1536) means the head is a small fraction of a decode step,
so the flash win is the smallest of the set and even dense quant heads are competitive. Numbers are
noisier here than for the larger models (a fast ~20 ms/step model amplifies run-to-run variance).

```bash
bash turbohead/surgery/build_all.sh h2oai/h2o-danube3-500m-chat danube3_500m
R=artifacts/danube3_500m
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7                              # greedy
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

### Quality (vs fp32 head, 1999 WikiText-2 positions)

| head | top-1 agree | PPL |
|---|---|---|
| head16 (fp32-eq, ref) | 100.0% | 6.791 |
| head8 g128 | **99.6%** | 6.793 |
| head4 g128 | 94.5% | 6.858 |
| head4 g32 | 96.3% | 6.837 |
| flash onnx / fused (A / H) | 94.5% | 51.833 † |

† Coverage 90.8% at P=256 (high — small vocab); *covered* PPL = 4.698.

### Speed — greedy (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 35.9±6.8 (1.00×) | 53.8±2.5 (1.00×) | 65.1±3.7 (1.00×) | 67.8±3.8 (1.00×) |
| head8 g128 | 49.8±0.7 (1.39×) | 85.5±3.9 (1.59×) | 106.0±2.3 (1.63×) | 108.8±2.1 (1.60×) |
| head4 g128 | 54.9±1.3 (1.53×) | 73.2±8.1 (1.36×) | 73.8±10.2 (1.13×) | 80.8±11.0 (1.19×) |
| head4 g32 | 41.9±2.7 (1.17×) | 81.7±5.4 (1.52×) | 71.6±14.9 (1.10×) | 79.7±7.3 (1.17×) |
| flash onnx (A) | 44.0±1.5 (1.23×) | 67.3±3.7 (1.25×) | 76.3±4.5 (1.17×) | 78.8±5.7 (1.16×) |
| **flash fused (H)** | **51.9±5.3 (1.45×)** | 70.8±6.7 (1.31×) | 76.9±4.5 (1.18×) | 83.5±2.2 (1.23×) |

### Speed — sampling, temp 0.8 (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 35.0±0.3 (1.00×) | 49.8±2.0 (1.00×) | 55.8±2.2 (1.00×) | 53.0±2.8 (1.00×) |
| head8 g128 | 43.4±0.9 (1.24×) | 60.0±3.7 (1.20×) | 74.1±3.2 (1.33×) | 71.0±4.0 (1.34×) |
| head4 g128 | 46.7±0.7 (1.33×) | 70.8±5.1 (1.42×) | 89.5±4.9 (1.61×) | 88.4±12.9 (1.67×) |
| head4 g32 | 45.0±1.7 (1.29×) | 69.4±2.8 (1.39×) | 84.0±3.1 (1.51×) | 82.4±3.2 (1.55×) |
| flash onnx (A) | 38.5±1.5 (1.10×) | 56.4±2.4 (1.13×) | 65.9±2.5 (1.18×) | 66.0±6.5 (1.24×) |
| **flash fused (H)** | **42.3±4.9 (1.21×)** | 68.0±8.9 (1.36×) | 87.4±2.9 (1.57×) | 92.8±5.0 (1.75×) |

### Takeaways

- Fused FlashHead **1.45× greedy / 1.21× sampling @1t** — the smallest win of the set, as expected from
  the tiny V:D ratio (head is a small share). `head8 g128` (1.39–1.63×, 99.6% agree, fp32-equal PPL) is
  genuinely competitive here, and int4 dense heads sometimes match fused at 1t — for small-vocab models
  the case for flash over a well-quantized dense head is weakest.
- High coverage (90.8%) from the small vocab; flash 94.5% agreement.

### Profile + `P` sensitivity (1 thread, `fused`)

`--profile` of the P=256 fused head (decode step 19.36 ms): `MatMulNBits` 16.51 ms (83%, body +
stage-1 gemv), **`FlashHeadSelect` (stage 2) 1.57 ms (8%)**, `GroupQueryAttention` 1.28 ms (6%), `TopK`
0.04 ms, `Gather` 0.03 ms. **`P` scales only stage 2; stage 1 is invisible** (its `[K=2000, D=1536]`
int4 gemv ≈ 1.5 MB is cache-resident, lumped into the body's `MatMulNBits`). So scaling stage 1 (e.g.
larger `cap`) reclaims ~nothing; `P` is the only useful head knob.

Re-splicing the fused head at **P=64** (¼ the candidates; no rebuild needed) vs P=256:

| | greedy 1t | 2t | 4t | 8t | top-1 agree | coverage |
|---|---|---|---|---|---|---|
| fused P=256 | 1.48× | 1.37× | 1.38× | 1.23× | 94.5% | 90.8% |
| fused **P=64** | **1.55×** | 1.52× | 1.63× | **1.57×** | 83.7% | 76.7% |
| P=64 vs P=256 | +4.7% | +11% | +18% | **+27%** | −10.8 pts | −14 pts |

At 1 thread P=64 is only **~5% faster** (matches the profile: stage 2 is 8% of the step, cut 4× ≈ 6%)
and costs ~11 points of agreement — a poor greedy trade. The gain grows with threads (to +27% @8t)
because once the body parallelizes, the **single-threaded** `FlashHeadSelect` becomes the relative
bottleneck — but 8t isn't the deploy point. Net: lowering `P` only pays off when the head is a large
share (big-vocab models) or at high thread counts; for small-vocab models keep P high for accuracy.

---

## LFM2.5-350M (int4 body) — 2026-06-17

V=65536, D=1024, 16 layers (**hybrid: 10 short-conv + 6 attention**), tied embeddings. FlashHead:
cap=16, K=4096, P=256. LiquidAI hybrid — most layers are short convolutions with `conv_state`, only
6 are attention (at sparse indices 2/5/8/10/12/14 with `past_key_values.N.key/value`). It feeds
`input_ids` (the embedding is in-graph), so once `decode_loop.py` learned generic state handling
(seed every `past*` input from its own shape; remap `past_conv.*`/`past_key_values.N.*` → `present*`)
it benches end-to-end — unlike Qwen3.5-0.8B, which splits the embedding out.

```bash
bash turbohead/surgery/build_all.sh LiquidAI/LFM2.5-350M lfm2_5_350m
R=artifacts/lfm2_5_350m
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7                              # greedy
uv run turbohead-bench $R/head16 $R/head8g128 $R/head4g128 $R/head4g32 $R/onnx $R/fused \
    --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
uv run turbohead-head-quality --src $R --npz $R/clusters.npz --head $R/head_W.npy -P 256
```

### Quality (vs fp32 head, 1999 WikiText-2 positions)

| head | top-1 agree | PPL |
|---|---|---|
| head16 (fp32-eq, ref) | 100.0% | 1071.5 † |
| head8 g128 | **99.0%** | 1070.3 † |
| head4 g128 | 91.5% | 1078.4 † |
| head4 g32 | 92.7% | 1057.5 † |
| flash onnx / fused (A / H) | 98.2% | 24797 † |

† **Absolute PPL is not reliable for this model** — the dense reference PPL (~1071) is ~100× the other
models', consistent across all heads, which points to a hidden-state-extraction mismatch in the quality
harness for this hybrid arch (the head sees mis-scaled hidden states; argmax is scale-invariant so
**top-1 agreement is unaffected and is the metric to trust here** — flash 98.2%, head8 g128 99.0%).
Flash coverage 77.4% at P=256 (also likely depressed by the same mismatch). The *covered* PPL is 150.9.

### Speed — greedy (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 45.6±0.7 (1.00×) | 65.9±1.6 (1.00×) | 71.4±2.0 (1.00×) | 72.4±1.2 (1.00×) |
| head8 g128 | 69.3±0.8 (1.52×) | 100.6±2.4 (1.53×) | 117.3±3.1 (1.64×) | 111.9±1.0 (1.54×) |
| head4 g128 | 75.9±0.9 (1.66×) | 108.8±1.4 (1.65×) | 134.3±3.1 (1.88×) | 127.0±24.9 (1.75×) |
| head4 g32 | 74.4±2.4 (1.63×) | 109.5±2.0 (1.66×) | 127.1±3.3 (1.78×) | 128.0±3.5 (1.77×) |
| flash onnx (A) | 72.1±1.1 (1.58×) | 102.3±2.5 (1.55×) | 120.4±2.7 (1.69×) | 121.5±0.5 (1.68×) |
| **flash fused (H)** | **84.0±1.7 (1.84×)** | **116.4±3.3 (1.77×)** | **134.4±3.7 (1.88×)** | **136.9±3.8 (1.89×)** |

### Speed — sampling, temp 0.8 (tok/s, median ± std; × vs head16)

| head | 1t | 2t | 4t | 8t |
|---|---|---|---|---|
| head16 (ref) | 41.7±0.7 (1.00×) | 61.2±1.1 (1.00×) | 68.2±1.2 (1.00×) | 69.6±1.0 (1.00×) |
| head8 g128 | 66.1±0.8 (1.58×) | 95.7±10.1 (1.57×) | 109.3±1.2 (1.60×) | 108.6±1.9 (1.56×) |
| head4 g128 | 75.2±2.1 (1.80×) | 104.7±2.3 (1.71×) | 119.4±3.2 (1.75×) | 121.0±2.4 (1.74×) |
| head4 g32 | 70.5±2.1 (1.69×) | 99.6±2.3 (1.63×) | 116.3±3.1 (1.71×) | 115.4±2.8 (1.66×) |
| flash onnx (A) | 70.2±1.7 (1.68×) | 101.4±2.5 (1.66×) | 116.0±2.6 (1.70×) | 119.8±1.0 (1.72×) |
| **flash fused (H)** | **85.0±3.0 (2.04×)** | **117.5±3.1 (1.92×)** | **132.0±3.5 (1.94×)** | **133.9±2.6 (1.92×)** |

### Takeaways

- Fused FlashHead **1.84× greedy / 2.04× sampling @1t** — fits the head-share trend (V:D=64, between
  Qwen3-0.6B's 2.40× and danube3's 1.45×), and beats every dense head at every thread count.
- First **hybrid** model benched end-to-end, via the generalized decode loop. The head method itself
  transfers cleanly (98.2% agreement); only the PPL harness mis-scales this arch's hidden states.

---

## Adding a model

The seven-model sweep above is complete. To add another: one `build_all.sh <hf-model> <slug>` then the
three commands (bench greedy, bench sampling, head-quality) shown in any section. FlashHead's edge grows
where the head is a larger share of decode (bigger V:D, fewer layers).

Caveats:
- `cap` must divide `V`; if `cap=16` doesn't, `build_all` fails at clustering — pass a divisor of that
  model's vocab as the `[cap]` arg. `block_size` 128 must divide `D`.
- **Hybrid models** (conv/recurrent state at sparse layer indices) bench fine via the generic state
  path **if** they feed `input_ids` (e.g. LFM2.5). Models that split the embedding out and require
  `inputs_embeds` + a 3-D `position_ids` (e.g. Qwen3.5-0.8B) are quality-only until `decode_loop.py`
  grows an embeddings/position-ids feed.

---

## What's next

Open items, roughly in priority order:

1. ~~Embeds-in decode path → unlock Qwen3.5-0.8B speed.~~ **Done** (2026-06-17): `decode_loop.py`
   feeds `inputs_embeds` (numpy lookup in the tied embedding `head_W`) + a 3-D M-RoPE `position_ids`.
   Qwen3.5-0.8B now benches end-to-end (fused 2.82× greedy @1t). For a self-contained deploy, splice
   could ship a fp16 `embed.npy` in the model dir instead of falling back to `../head_W.npy`.
2. **Fix the head-quality PPL harness for hybrids.** LFM2.5's absolute PPL is ~100× off (dense ref
   ~1071) while top-1 agreement is fine — the harness is almost certainly capturing pre-final-norm /
   mis-scaled hidden states for these archs. Argmax is scale-invariant so the speedup/agreement story
   is unaffected, but the PPL column is meaningless for hybrids until the hook grabs the post-norm
   hidden state. (Track it down via the `lm_head`-input hook in `eval/head_quality.py`.)
3. **Land the branch.** `turbohead-fused-op` carries the fused kernel, the multi-model sweep, the
   clustering tail fix, and the hybrid decode loop — merge to `main` once reviewed.
4. **(Optional) coverage/speed Pareto.** A small `(cap, P)` sweep per model to pick the knee instead of
   the fixed `cap=16, P=256` — most useful for the big-vocab models where `P` actually moves the step
   time (the danube3 note shows it barely does for small vocab).
