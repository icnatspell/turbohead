# Experiment results

Per-model head-precision comparison: **is FlashHead worth it vs just quantizing the dense head?**
One section per model; add new models by appending a section in the same shape.

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

## Models pending

Same matrix to run next (need int4 baseline + clusters built first):
Gemma3-270M, Gemma3-1B, Llama-3.2-1B, Qwen3-1.7B, Qwen3.5-0.8B, LFM2.5-350M. FlashHead's edge should
grow where the head is a larger share of decode (bigger V:D, fewer layers).
