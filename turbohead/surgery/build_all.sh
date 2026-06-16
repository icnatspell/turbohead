#!/usr/bin/env bash
# Build the full head-comparison artifact set for one HF model, into its own artifacts/<slug>/ dir
# (reusable — nothing shared between models). Reproducible end-to-end: genai int4 baseline -> head
# weight -> clusters -> 4 dense-head variants + 2 flash splices (onnx contract-A, fused contract-H).
# Then bench/eval with the commands printed at the end.
#
#   bash turbohead/surgery/build_all.sh <hf-model> <slug> [cap] [P]
#   bash turbohead/surgery/build_all.sh Qwen/Qwen3-1.7B qwen3_1_7b
#
# Layout: artifacts/<slug>/{baseline, head_W.npy, clusters.npz, head16, head8g128, head4g128,
#         head4g32, onnx, fused}.
# Idempotent: re-running skips the (slow) genai baseline if already built; set FORCE=1 to rebuild.
set -euo pipefail

MODEL=${1:?usage: build_all.sh <hf-model> <slug> [cap] [P]}
SLUG=${2:?usage: build_all.sh <hf-model> <slug> [cap] [P]}
CAP=${3:-16}
P=${4:-256}
ROOT=artifacts/$SLUG
BASE=$ROOT/baseline
HEAD=$ROOT/head_W.npy
NPZ=$ROOT/clusters.npz

echo "== build_all: $MODEL -> $ROOT/ (cap=$CAP P=$P) =="

[ -f csrc/libturbohead.so ] || bash csrc/build.sh           # fused-op kernel (one-time)

if [ "${FORCE:-0}" = 1 ] || [ ! -f "$BASE/model.onnx" ]; then
  rm -rf "$BASE"
  MODEL="$MODEL" OUT="$BASE" CACHE="artifacts/hf_cache" bash turbohead/surgery/convert_baseline.sh
else
  echo "-- baseline exists ($BASE/model.onnx); skip (FORCE=1 to rebuild)"
fi

uv run turbohead-extract-head   --model "$MODEL" --out "$HEAD"
uv run turbohead-build-clusters --head "$HEAD" --out "$NPZ" --cap "$CAP"

# dense-head baselines (the comparison points): fp32-eq, int8, int4 at two group sizes
uv run turbohead-quantize-head --src "$BASE" --head "$HEAD" --bits 16                 --dst "$ROOT/head16"
uv run turbohead-quantize-head --src "$BASE" --head "$HEAD" --bits 8  --group-size 128 --dst "$ROOT/head8g128"
uv run turbohead-quantize-head --src "$BASE" --head "$HEAD" --bits 4  --group-size 128 --dst "$ROOT/head4g128"
uv run turbohead-quantize-head --src "$BASE" --head "$HEAD" --bits 4  --group-size 32  --dst "$ROOT/head4g32"

# flash heads: portable onnx (contract A) + fused custom op (contract H)
uv run turbohead-splice --backend onnx  --src "$BASE" --npz "$NPZ" --head "$HEAD" -P "$P" --dst "$ROOT/onnx"
uv run turbohead-splice --backend fused --src "$BASE" --npz "$NPZ" --head "$HEAD" -P "$P" --dst "$ROOT/fused"

cat <<EOF

== built. now bench + eval ==
  uv run turbohead-bench $ROOT/head16 $ROOT/head8g128 $ROOT/head4g128 $ROOT/head4g32 \\
      $ROOT/onnx $ROOT/fused --threads 1,2,4,8 --reps 7
  uv run turbohead-bench ... --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0   # sampling
  uv run turbohead-head-quality --src $ROOT --npz $NPZ --head $HEAD -P $P
EOF
