# whisper_head

**Status: BUILT + BENCHED (2026-06-22) on `whisper-tiny/base/small`. FlashHead transfers to
encoder-decoder ASR: flash-fused (int8 stage-2) beats an fp32/fp16 dense head 1.55×–2.34× (scales with
head share), beats int8 1.14×–1.24×, ≈ par with int4; top-1 agreement 94–97% @ P=256. The genai int4
head is the hard floor — flash's win is bytes-saved = head-precision × head-share, not head-share alone.**

## What this is

End-to-end FlashHead on Whisper. The head lives in the genai-exported `decoder.onnx` (emits a `logits`
node, so `splice.py::find_head` works unchanged); cross-attn KV is precomputed by the encoder and fed
as constant decoder inputs, self-attn KV grows. The surgery core is reused as-is — only Whisper's
encoder-decoder runtime + head extraction are new here.

## Files

- `whisper_head_poc.py` — torch/HF gate: head share of a decode step + top-1/top-5 agreement on real
  decoder hidden states (transcribing librispeech-dummy). Run per `--model`.
- `build_whisper.py` — one model end to end: genai int4 export → extract `proj_out` → cluster →
  splice (onnx+fused) → 4 dense-head precision baselines (fp32 / fp16=head16 / int8 / int4 via core
  `quantize_head`). `uv run python experimental/whisper_head/build_whisper.py openai/whisper-small whisper_small`
- `whisper_decode.py` — raw-ORT encoder-decoder decode loop + per-step bench (the harness
  `decode_loop.py` doesn't cover: cross-KV has no `present`, int32 ids, encoder pass, no attn mask).
- `sweep.py` — benches all 6 head variants × all built models into one table (one subprocess per
  variant — two custom-op `.so` in a process segfault, same as `turbohead-bench`).

## Results (2026-06-22, int4 body, CPU, 1 thread, P=256, cap=41 → K=1265)

### Model shape & head share of a decode step (the Amdahl ceiling)

| model | D | dec. layers | head share | top-1 agree @256 | top-5 agree @256 |
|---|---|---|---|---|---|
| whisper-tiny  | 384 | 4  | **67.8%** | 96.6% | 96.6% |
| whisper-base  | 512 | 6  | **51.3%** | 95.0% | 95.0% |
| whisper-small | 768 | 12 | **26.0%** | 94.5% | 94.5% |

Encoder runs once (amortized); share = head / (head + decoder-per-step), the memory-bound M=1 proxy.
Agreement = recall of the dense-fp32-head argmax on real hidden states (sweep `-P` 128→512 lifts it:
small goes 89.0 → 94.5 → 97.4 → 99.0%).

**top-5 == top-1 in every row, exactly.** Not a bug: FlashHead misses are *coverage* misses — when the
true token's cluster isn't in the top-P probed set it isn't scored at all, so it's absent from the whole
candidate list, not merely outranked. Covered tokens are scored exactly (stage-2 is a real dot product),
so a covered dense-argmax is *always* flash's top-1. ⇒ the accuracy lever is **P (coverage), not k**;
top-k beyond 1 buys nothing. (Raising P is the way to recover the last few %; see CLAUDE.md.)

### Speed — median decoder step (ms), 4 dense-head precisions vs the 2 flash backends

flash-fused uses **int8 stage-2** (`--head-weight-dtype int8`, `FlashHeadSelectQ8`); with the fp32
stage-2 default it reads *more* bytes than the int4 head and loses to it (tiny 0.84×) — see Findings.

| model | fp32 | fp16 | int8 | int4 | flash (onnx) | **flash-fused** |
|---|---|---|---|---|---|---|
| whisper-tiny  | 5.96 | 6.09 | 3.23 | 2.75 | 5.33 | **2.61** |
| whisper-base  | 8.82 | 8.86 | 5.45 | 4.96 | 8.31 | **4.74** |
| whisper-small | 20.51 | 19.96 | 14.76 | 12.73 | 17.52 | **12.91** |

### flash-fused speedup vs each dense-head precision (>1× = flash faster)

| model | vs fp32 | vs fp16 | vs int8 | vs int4 |
|---|---|---|---|---|
| whisper-tiny  | **2.28×** | 2.34× | 1.24× | 1.06× |
| whisper-base  | **1.86×** | 1.87× | 1.15× | 1.05× |
| whisper-small | **1.59×** | 1.55× | 1.14× | 0.99× |

(±several % run-to-run; the ≈1.0× int4 column shouldn't be over-read — small's int4 baseline swung
14.5→12.7 ms across runs. fp32/fp16/int8 are all clear wins.)

## Findings

- **The win is bytes-saved, = head-precision × head-share, NOT head-share alone.** Speed at M=1 is
  memory-bound. The head's *byte* share depends on its precision: tiny's head is 67.8% of *params* but
  79.6 MB at fp32 vs only 9.96 MB at int4 (body is int4 throughout). Flash-fused (int8 stage-2) reads
  `P·cap·D` ≈ 4 MB. So vs fp32 (80 MB) it's a huge cut → 2.28×; vs int4 (10 MB) there's almost nothing
  left → ≈par. **FlashHead and head-quantization optimize the same bottleneck (head bytes), so they
  don't stack** — once the head is int4 the bottleneck is already gone.
- **Beats fp32/fp16 1.55–2.34× and int8 1.14–1.24×, scaling with head share** (tiny's 67.8% → biggest).
  This is the RESULTS.md framing and the legitimate use case: a deploy whose head must stay fp16/fp32/int8
  (precision-sensitive, tied embeddings, or no fast int4 head kernel).
- **≈ par with int4** (0.99–1.06×). `MatMulNBits` at M=1 is a very cheap fused-dequant gemv; flash can't
  beat it by much because (a) the head is already tiny in bytes and (b) flash has an irreducible floor —
  the stage-1 routing gemv + TopK + gather of P·cap (20% of vocab) rows + exact scoring — which at these
  2–13 ms absolute latencies is a real fraction. Same Amdahl wall the core hits vs int8 (~1.33× ceiling).
- **int8 stage-2 is essential to even reach par vs int4.** With the fp32 stage-2 default, flash-fused
  reads `P·cap·D·4` ≈ 16 MB — *more* than the whole int4 head (10 MB) — and loses (tiny 0.84×). int8
  stage-2 (`FlashHeadSelectQ8`) quarters that to ~4 MB; build_whisper now defaults fused to int8.
- **`flash` (onnx backend) never wins** — it materializes full-V logits; only `flash-fused` (shortlist)
  is competitive, as in core.
- **fp16 ≈ fp32 (sometimes slower)** on CPU — no fp16 matmul kernel, folds to fp32 + a Cast (ORT_QUIRKS).

## Reproduce

```bash
uv sync --extra surgery
for m in tiny base small; do
  uv run python experimental/whisper_head/build_whisper.py openai/whisper-$m whisper_$m
  uv run python experimental/whisper_head/whisper_head_poc.py --model openai/whisper-$m   # agreement + head share
done
uv run python experimental/whisper_head/sweep.py                                          # speed table
```

## Promotion path

New *model target*, not a method lever. To graduate: fold the encoder-decoder feed path into core
`decode_loop.py` (cross-KV-from-encoder + int32 ids + mel front-end) and a Whisper branch in the export,
then add whisper-* to `build_all.sh` + the RESULTS sweep. Worth it for fp32/fp16/int8-head deploys
(1.14–2.34×); against an int4-everything deploy it's ≈par, so not compelling there.
