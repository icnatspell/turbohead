# int8 weights for the FlashHead stage-2 table (`fh_Wperm`)

The fused FlashHead is the fastest head on the 6-core CPU board. We made it faster by
storing its weight table as int8 instead of fp32, leaving the kernel logic alone. This doc
records what we measured and how to build it.

## What the fused head does

For each decoded token the head turns the final hidden vector `h` (length `D=1024`) into
logits, then picks the next token. A plain head multiplies `h` by the full vocabulary matrix
(`V=151936` rows), which costs a lot. FlashHead splits that into two stages:

```
stage 1   fh_sims = h · Cnorm        # score h against 9496 cluster centroids (cheap, 4-bit)
          TopK                        # keep the 256 best clusters
stage 2   FlashHeadSelect            # for those 256 clusters, score only their candidate
                                      # rows (256 x 16 = 4096 tokens, not all 151936)
```

Stage 2 is the custom C++ op. Per token it reads `256 clusters x 16 rows x 1024` weights from
a table called `fh_Wperm`. The build stored that table as fp32.

## What the profiling showed

Decode-step breakdown (board, single thread, from the logs that opened this session):

| bucket | ms/step | share | what it is |
|---|---|---|---|
| MatMulNBits | 98.6 | 70% | the 28 transformer layers (the model body) |
| GroupQueryAttention | 20.6 | 15% | attention |
| FlashHeadSelect | 6.0 | ~4% | stage 2, the head |
| other | ~14 | ~10% | norms, reshapes, etc. |

We acted on two findings:

1. The board ran single-threaded on a 6-core chip. The body (85% of the step: `MatMulNBits`
   plus attention) spreads across cores; the head does not. Pass `--threads 6` to use all
   cores. This is the biggest single win and costs nothing.
2. The head runs serial because it is memory-bound (see the comment in
   `csrc/turbohead_op.cc`). When the body spreads across 6 cores and shrinks, the head stays
   the same size and grows as a share of each step. So the head earns attention on a
   multi-core board.

We ruled out two other levers. `MatMulNBits` already runs `accuracy_level=4` (int8 math),
and the `Reshape` ops in the profile sit in the body, not the head.

## The change: store `fh_Wperm` as int8

Stage 2 reads weights and multiplies. The byte count of that read sets its speed, so we cut
the byte count.

| storage | bytes/token (stage 2) | full table size |
|---|---|---|
| fp32 | 16.8 MB | 622 MB |
| fp16 | 8.4 MB | 311 MB |
| **int8** | **4.2 MB** | **156 MB** |
| int4 | 2.1 MB | 78 MB |

We quantize per output channel, one scale per weight row: `scale = max(abs(row)) / 127`,
store the row as int8, and fold the scale back in once at the end of each dot product.

## Why int8 and not fp16 or int4

We microbenchmarked the stage-2 read pattern (random clusters across the full table, so the
cache behaviour matches reality) for each dtype. Numbers come from the dev workstation, an
x86 chip with more bandwidth than the board:

| dtype | ms/token | vs fp32 |
|---|---|---|
| fp32 | 0.800 | 1.0x |
| fp16 | 1.196 | 0.67x (slower) |
| **int8** | **0.445** | **1.8x faster** |
| int4 | 2.806 | 0.29x (much slower) |

Fewer bytes did not always mean faster:

- int8 wins. Converting int8 to float is one cheap step the compiler vectorizes.
- fp16 loses. CPUs run no native fp16 math, so each weight needs a software half-to-float
  conversion that eats the bandwidth saving. It would win only with hardware fp16 (ARM
  NEON-FP16) wired into the kernel.
- int4 loses by more. Unpacking two 4-bit values per byte blocks vectorization. It would
  need hand-written SIMD to pay off.

int8 fits a CPU best: fewer bytes and cheap to decode. The ranking shifts with a chip's
bandwidth-to-compute balance, so it may differ on the board, though int8 stays the safe CPU
pick and the one we shipped.

## Accuracy

No meaningful loss.

- Per-weight quantization error stays at 0.20% of each row's max magnitude.
- Greedy decode of a 64-token sample matched fp32 on 63 of 64 tokens. The one swap, "a
  beautiful beach" to "a great beach", came from a near-tied logit where rounding flipped the
  pick, and the text stayed fluent. Sampling buries a difference this size in its own noise.
- The model body already runs 4-bit (`Q4G128`). The 4-bit body sets the accuracy floor, and
  an 8-bit per-channel head sits well above it.

For a hard number, `src/blockrot/ppl.py` compares fp32-head against int8-head perplexity. We
expect the gap to land inside measurement noise.

## The two variants

| variant | model dir | head op | head weights |
|---|---|---|---|
| fp32 (baseline) | `artifacts/qwen3_0_6b/fused/` | `FlashHeadSelect` | fp32, 622 MB |
| int8 (new) | `artifacts/qwen3_0_6b/fused_q8/` | `FlashHeadSelectQ8` | int8, 156 MB |

You pick the op by which model dir you run. The splice step bakes the op into the model. The
decode loop registers both ops from the `.so` and runs whichever node the loaded model holds,
so no runtime flag and no rebuild when you switch dirs.

## Generate the model artifact

The int8 head comes from the normal splice pipeline with one extra flag,
`--head-weight-dtype int8`. Build the custom-op `.so` first, since the fused backend needs
it.

```bash
bash csrc/build.sh      # builds csrc/libturbohead.so with both ops (run on the target arch)
```

### From an existing artifact tree (the common case)

If you already built a model with `build_all.sh` (so `baseline/`, `head_W.npy`, and
`clusters.npz` exist), one splice gives you the int8 head:

```bash
R=artifacts/qwen3_0_6b
uv run turbohead-splice --backend fused --head-weight-dtype int8 \
  --src $R/baseline --npz $R/clusters.npz --head $R/head_W.npy -P 256 --dst $R/fused_q8
```

The fp32 head is the same command without the flag (the existing `$R/fused` dir):

```bash
uv run turbohead-splice --backend fused \
  --src $R/baseline --npz $R/clusters.npz --head $R/head_W.npy -P 256 --dst $R/fused
```

### From scratch (no artifact tree yet)

Run the four pipeline steps, then the int8 splice. Steps 1 to 3 match the README; step 4
adds the flag:

```bash
R=artifacts/qwen3_0_6b
# 1. int4 baseline ONNX (genai model builder)
MODEL=Qwen/Qwen3-0.6B OUT=$R/baseline bash turbohead/surgery/convert_baseline.sh
# 2. dump the fp32 head weight (= tied embedding)
uv run turbohead-extract-head   --model Qwen/Qwen3-0.6B --out $R/head_W.npy
# 3. balanced k-means clustering assets (cap=16 -> K = V/cap clusters)
uv run turbohead-build-clusters --head $R/head_W.npy --out $R/clusters.npz --cap 16
# 4. splice the fused head with int8 stage-2 weights
uv run turbohead-splice --backend fused --head-weight-dtype int8 \
  --src $R/baseline --npz $R/clusters.npz --head $R/head_W.npy -P 256 --dst $R/fused_q8
```

`fp32` writes a `FlashHeadSelect` node with an fp32 `fh_Wperm`. `int8` writes a
`FlashHeadSelectQ8` node with an int8 `fh_Wperm_q8` plus per-channel `fh_Wperm_scale`.
`fused_stage2_nodes` in `build_subgraph.py` does the quantizing, the one place that writes
fused head weights. The splice step also copies `libturbohead.so` into the output dir.

## How to test on the board

The board runs ARM, so the x86 `.so` from the workstation will not load there. Rebuild on
the board:

```bash
# in flashhead-ort/, with both model dirs present:
bash csrc/build.sh                                   # builds the ARM .so (both ops)
cp csrc/libturbohead.so artifacts/qwen3_0_6b/fused/
cp csrc/libturbohead.so artifacts/qwen3_0_6b/fused_q8/
for d in fused fused_q8; do
  python -m turbohead.inference.decode_loop artifacts/qwen3_0_6b/$d --threads 6 --profile
done
```

Compare the two `tok/s` lines and the `FlashHeadSelect` against `FlashHeadSelectQ8` rows in
the profile. Expect matching accuracy and a smaller, faster head. The gap should widen on the
board, since the head holds a bigger serial share once the body fills 6 threads.

If `fused_q8/` is not on the board yet, re-run the `turbohead-splice ... int8` command there,
or copy the `fused_q8/model.onnx*` files across (about 554 MB).

## What changed in the codebase

- `csrc/turbohead_op.cc` adds the `FlashHeadSelectQ8` op next to the fp32 `FlashHeadSelect`.
  `csrc/build.sh` compiles this file into the `.so`.
- `turbohead/surgery/build_subgraph.py`: `fused_stage2_nodes` takes a `weight_dtype`
  argument that writes the int8 node and quantized weights.
- `turbohead/surgery/splice.py` exposes it as `--head-weight-dtype {fp32,int8}`.

`csrc/build.sh` needs no change. It already compiles `turbohead_op.cc`.
