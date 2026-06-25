# Head quantization configs

Build matrix for the FlashHead two-stage head: **1st matrix** = stage-1 centroid scoring
(`h · Cᵀ`), **2nd matrix** = stage-2 candidate-row dot. Each can be quantized independently, on
either backend (`onnx` = unfused / portable, `fused` = custom op). `g` = per-group, `ch` =
per-channel.

## Flag legend

| precision | 1st matrix (`--stage1` / `--block-size`) | 2nd matrix (`--head-weight-dtype`) |
|---|---|---|
| **FP**     | `--stage1 fp16`                          | `--head-weight-dtype fp32` |
| **INT8g**  | `--stage1 int8 --block-size 128`         | — *(stage 2 has no per-group int8)* |
| **INT8ch** | `--stage1 int8ch`                        | `--head-weight-dtype int8` |
| **INT4g**  | `--stage1 int4 --block-size 128`         | — *(stage 2 has no int4)* |

Backend: `--backend onnx` (unfused) or `--backend fused`.

**Which op does what, and why it matters:**
- **Per-group int8/int4 (1st)** → `MatMulNBits` (fused dequant in the gemv, wins at M=1). `--block-size`
  is the group size; **max 256** — MLAS silently computes garbage above that (docs/ORT_QUIRKS.md).
- **Per-channel int8 (1st, `int8ch`)** → `MatMulInteger` + per-channel scale `Mul` (the canonical ONNX
  per-channel path — *not* MatMulNBits, which is block-wise). Correct, but **slower at M=1** than
  per-group MatMulNBits (no fused dequant). Per-group int8 is both finer-grained and faster, so
  `int8ch` is here to *measure* per-channel, not because it's a good operating point for stage 1.
- **Per-channel int8 (2nd)** → fused `FlashHeadSelectQ8` (dots straight from int8), or the onnx int8
  rows + hoisted-scale `Mul`. No MatMulNBits, no block limit. **Stage 2 supports only FP and INT8ch.**

## The configs

|  # | backend | 1st | 2nd | extra flags |
|---:|---|---|---|---|
|  1 | onnx (unfused)  | INT4g  | INT8ch | `--stage1 int4 --block-size 128` `--head-weight-dtype int8` |
|  2 | onnx (unfused)  | INT8ch | INT8ch | `--stage1 int8ch`                `--head-weight-dtype int8` |
|  3 | onnx (unfused)  | INT8ch | FP     | `--stage1 int8ch`                `--head-weight-dtype fp32` |
|  4 | fused           | INT4g  | INT8ch | `--stage1 int4 --block-size 128` `--head-weight-dtype int8` |
|  5 | fused           | INT8ch | INT8ch | `--stage1 int8ch`                `--head-weight-dtype int8` |
|  6 | fused           | INT8ch | FP     | `--stage1 int8ch`                `--head-weight-dtype fp32` |

## Build commands

Assumes a built model (`bash src/turbohead/surgery/build_all.sh Qwen/Qwen3-0.6B qwen3_0_6b`).
`fused` configs (4–6) need `bash csrc/build.sh` first.

```bash
R=artifacts/qwen3_0_6b
S="--src $R/baseline --npz $R/clusters.npz --head $R/head_W.npy -P 256"

# 1  unfused | 1st INT4g  | 2nd INT8ch
uv run turbohead-splice --backend onnx  $S --stage1 int4 --block-size 128 --head-weight-dtype int8 --dst $R/cfg1
# 2  unfused | 1st INT8ch | 2nd INT8ch
uv run turbohead-splice --backend onnx  $S --stage1 int8ch                --head-weight-dtype int8 --dst $R/cfg2
# 3  unfused | 1st INT8ch | 2nd FP
uv run turbohead-splice --backend onnx  $S --stage1 int8ch                --head-weight-dtype fp32 --dst $R/cfg3
# 4  fused   | 1st INT4g  | 2nd INT8ch
uv run turbohead-splice --backend fused $S --stage1 int4 --block-size 128 --head-weight-dtype int8 --dst $R/cfg4
# 5  fused   | 1st INT8ch | 2nd INT8ch
uv run turbohead-splice --backend fused $S --stage1 int8ch                --head-weight-dtype int8 --dst $R/cfg5
# 6  fused   | 1st INT8ch | 2nd FP
uv run turbohead-splice --backend fused $S --stage1 int8ch                --head-weight-dtype fp32 --dst $R/cfg6
```

Measure: `uv run turbohead-bench $R/cfg1 $R/cfg2 $R/cfg3 $R/cfg4 $R/cfg5 $R/cfg6 --threads 1 --reps 7`
(one model per subprocess — two custom-op `.so` in one process segfault).

## Notes (docs/ORT_QUIRKS.md)

- On the **onnx** backend, 2nd=INT8ch tends to land flat-to-negative on CPU EP: the `Cast` back to fp32
  re-materializes the rows it just saved reading. The int8 stage-2 win only shows up in `fused`.
- `int8ch` (1st) is expected **slower** than `int4`/`int8` per-group — `MatMulInteger` has no fused
  dequant. Build it to fill the matrix; don't expect it to win.
- Any other 1st×2nd combo is just mixing the legend flags (within the supported set above).
