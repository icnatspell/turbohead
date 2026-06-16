# ONNXRuntime CPU operator quirks (TurboHead findings)

Hard-won, **measured** quirks of the ONNXRuntime CPU EP (version **1.26.0**) hit while
optimizing TurboHead's approximate LM head. Each entry: what we expected, what actually
happened, and the takeaway. The recurring theme — **the CPU EP has fast kernels only for
the shapes/dtypes it was tuned for (static-weight matmuls, fp32 gemm); step off that path
and you fall back to something slow or an auto-inserted cast.**

> Methodology note up top, because it bit us twice: **a standalone microbenchmark of an op
> does NOT predict its cost in the full model.** Graph-level transforms (fusion, precision-cast
> insertion) only fire in the real graph, and cache/bandwidth contention with the rest of the
> model changes the picture. **Always confirm an op-level change by re-splicing and profiling
> the real model, and change ONE thing per measurement.**

---

## 1. No fp16 MatMul / FusedMatMul kernel on CPU — fp16 inputs get cast to fp32

**Expected:** storing weights fp16 halves gather+matmul bandwidth → faster at M=1.
**Actual:** ORT inserts `InsertedPrecisionFreeCast_*` casting the fp16 operand back to fp32
before the matmul, because the CPU `MatMul`/`FusedMatMul` kernel is fp32-only. The cast
(~1.6ms for our P·cap·D rows) cancels the bandwidth saving — net **slower** end-to-end.

- Confirmed for both plain `MatMul` and `FusedMatMul` (the Transpose-fused variant).
- A standalone microbench showed fp16 *faster* (it didn't trigger the cast insertion the full
  graph does) — see the methodology note.
- **Takeaway:** keep stage-2 weights **fp32** on CPU. fp16 only pays off where there's a real
  fp16 kernel (GPU, or specific contrib ops).

## 2. MatmulTransposeFusion is fp32-only

**Expected:** `Transpose(h) → MatMul` fuses into the optimized `com.microsoft.FusedMatMul`.
**Actual:** it does — but `FusedMatMul` is fp32-only, so with fp16 inputs you get quirk #1.
Replacing `Transpose` with a `Reshape` (free, same data for a vector) dodges the fusion, but
the resulting plain `MatMul` is *also* fp32-only — the cast still gets inserted.
**Takeaway:** you can't escape the fp32 matmul on CPU by restructuring around the fusion.

## 3. No MatMulNBits-equivalent for *dynamically-gathered* rows

**Expected:** quantize stage-2 weights (int8/int4) to cut the dominant gather bandwidth.
**Actual:** every fast quantized matmul ORT ships — `MatMulNBits`, `DynamicQuantizeMatMul`,
`MatMulInteger` — requires the quantized weight as a **static, pre-packed initializer**. When
*which rows* you touch is decided at runtime (here by a TopK), you can't use any of them. You're
forced onto generic `Gather` + `DequantizeLinear` + generic `MatMul`, and the dequant pass writes
back exactly the fp32 bytes you were trying to avoid. Measured: int8-gather+dequant ≈ fp16 ≈ no
better than fp32. **Takeaway:** dynamic-gather + quantized-matmul fusion does not exist on CPU;
the only way to fuse gather+dot is a custom kernel.

## 4. `GatherBlockQuantized` CPU kernel is ~20× too slow for large gathers

**Expected:** the one op that fuses gather+dequant (int4 data + fp scales) — built for tied
embeddings — could serve as a fused quantized stage-2.
**Actual:** runs on CPU, bit-exact, but **50ms vs 2.4ms** for a fp32 `Gather` of the same
P·cap=4096 rows. Its kernel is tuned for embedding lookups (a handful of rows) with scalar
per-row dequant; it does not scale to thousands of gathered rows.
**Takeaway:** correct ≠ fast. Profile contrib ops at your actual shapes before relying on them.

## 5. `MatMulNBits` (int4/int8) *wins big* at M=1 — unlike other int matmuls

**Expected (from earlier work):** "int8 anything is slower than fp16 at M=1 on CPU."
**Actual:** true for `MatMulInteger` and manual dequant (ORT's CPU int8 gemv loses to its tuned
fp16 gemv at M=1), but **`MatMulNBits` is the opposite** — W4A16/W8A16 with `accuracy_level=4`,
it fuses dequant into the gemv and runs ~9× faster than fp16 on the M=1 centroid-scoring gemv,
with 100% argmax agreement. It's the same op the quantized model body uses.
**Takeaway:** for any **static-weight** int gemv at M=1, use `MatMulNBits`, never `MatMulInteger`.
(Note this is the static-weight case — contrast with #3, the dynamic-gather case, where it can't be used.)

## 6. The O(V) `ScatterElements` is cheap; the full-vocab softmax is not

**Expected:** scattering candidate logits into a `(1,V)` `-1e9` base (V≈152k) every step is costly.
**Actual:** `ScatterElements` is ~0.1ms — negligible. What's expensive is a **full-vocab softmax**
for sampling (~2.2ms over V) — which you only pay if you softmax the whole `(1,V)`. Softmaxing
just the scored candidates (≈4k) is ~0.05ms.
**Takeaway:** don't fear the O(V) scatter; do avoid the O(V) softmax — sample over the candidate
shortlist, not the full vocab. (Contract A's decode loop already does this via `row > -1e8`.)

## 7. Profiling inflates `model_run` — don't mix profiled and unprofiled timings

**Observed:** profiled decode `model_run` median was ~23ms while the unprofiled tok/s implied
~20ms/step. ORT's `enable_profiling` instrumentation adds per-op overhead to every run.
**Takeaway:** compare **profiled-to-profiled** (e.g. two variants both profiled) for per-op
attribution, and use **unprofiled tok/s** (median of several reps) for headline speed. A single
unprofiled run is pure noise on this box (±3–5 tok/s); always take a median, and interleave
variants in the same session when comparing.

## 8. Parallelizing a memory-bound custom op doesn't help (and can hurt)

**Expected:** OpenMP over the per-cluster loop scales the fused stage-2 across cores.
**Actual (controlled, same kernel serial vs OpenMP):** 1t 53.8/53.2, 4t 83.4/84.0 (noise),
8t **83.0/77.1** — no gain at 1–4 threads, ~7% *slower* at 8. The loop streams ~16.8MB of weight
rows: it's bandwidth-bound, so one thread already saturates the relevant DRAM bandwidth, and extra
threads add fork/join cost and oversubscribe against ORT's spinning idle intra-op threads.
**Takeaway:** parallelize compute-bound kernels, not bandwidth-bound ones. The ORT intra-op
threadpool already parallelizes the heavy (compute-bound) body matmuls; a small memory-bound head
kernel should stay serial. Also beware nested threadpools (OpenMP threads + ORT spinning threads).

---

## Custom-op build quirks (ORT 1.26 lite custom-op API)

For a CPU custom op via `SessionOptions.register_custom_ops_library` (see `csrc/`):

- **No headers in the pip wheel.** `onnxruntime_c_api.h` / `onnxruntime_cxx_api.h` / lite-op headers
  must be fetched from the matching `v<VER>` git tag. They pull transitive includes
  (`onnxruntime_ep_c_api.h`, …) — resolve them iteratively. See `csrc/build.sh`.
- **`Ort::InitApi` requires `#define ORT_API_MANUAL_INIT`** before including the C++ API header;
  a custom-op lib sets the global API pointer itself in `RegisterCustomOps`.
- **`OrtLiteCustomOp` lives in `Ort::Custom::`**, not the global namespace.
- **`RegisterCustomOps`** must be `extern "C"`, signature
  `OrtStatus*(OrtSessionOptions*, const OrtApiBase*)`; the C API
  (`CreateCustomOpDomain`/`CustomOpDomain_Add`/`AddCustomOpDomain`) is the version-robust path.
- **`-ffast-math` is required** for a `float +=` dot-product reduction to auto-vectorize; strict FP
  ordering otherwise forces a scalar loop (~2× slower). Build `-O3 -march=native -ffast-math`.
- A custom op can take large initializers (e.g. the full weight matrix) as inputs **by reference** —
  ORT passes a pointer, no copy. This is what lets the fused kernel read only the gathered rows.
