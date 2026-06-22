#!/usr/bin/env bash
# Build the full head-comparison artifact set for one HF model, into its own artifacts/<slug>/ dir
# (reusable — nothing shared between models). Reproducible end-to-end: genai int4 baseline -> head
# weight -> clusters -> 4 dense-head variants + 2 flash splices (onnx logits-out, fused shortlist-out).
# Then bench/eval with the commands printed at the end.
#
#   bash src/turbohead/surgery/build_all.sh <hf-model> <slug> [cap] [P]
#   bash src/turbohead/surgery/build_all.sh Qwen/Qwen3-1.7B qwen3_1_7b
#
# SKIP_DENSE=1 skips the 4 dense-head baselines (the comparison points) and builds only the flash
# splices + their inputs — the actual product. Used by the CI end-to-end smoke test.
#
# Layout: artifacts/<slug>/{baseline, head_W.npy, clusters.npz, head16, head8g128, head4g128,
#         head4g32, onnx, fused}.
# Idempotent: re-running skips the (slow) genai baseline if already built; set FORCE=1 to rebuild.
# Optional: ETA_SWEEP="2 4" runs a per-model ScaNN-anisotropy sweep (agreement-gated) before splicing.
set -euo pipefail

MODEL=${1:?usage: build_all.sh <hf-model> <slug> [cap] [P]}
SLUG=${2:?usage: build_all.sh <hf-model> <slug> [cap] [P]}
CAP=${3:-16}
P=${4:-256}
ROOT=artifacts/$SLUG
BASE=$ROOT/baseline
HEAD=$ROOT/head_W.npy
NPZ=$ROOT/clusters.npz
LOGS=$ROOT/logs

mkdir -p "$LOGS"
exec > >(tee "$LOGS/build.log") 2>&1   # keep the full build log with the artifacts

echo "== build_all: $MODEL -> $ROOT/ (cap=$CAP P=$P) =="
echo "$MODEL" > "$ROOT/hf_model_id.txt"   # makes the artifact dir self-describing (head_quality reads it)

[ -f csrc/libturbohead.so ] || bash csrc/build.sh           # fused-op kernel (one-time)

if [ "${FORCE:-0}" = 1 ] || [ ! -f "$BASE/model.onnx" ]; then
  rm -rf "$BASE"
  MODEL="$MODEL" OUT="$BASE" CACHE="artifacts/hf_cache" bash src/turbohead/surgery/convert_baseline.sh
else
  echo "-- baseline exists ($BASE/model.onnx); skip (FORCE=1 to rebuild)"
fi

# head_W + clusters are deterministic and slow (clustering ~minutes); reuse unless FORCE=1.
if [ "${FORCE:-0}" = 1 ] || [ ! -f "$HEAD" ]; then uv run turbohead-extract-head   --model "$MODEL" --out "$HEAD"; fi
if [ "${FORCE:-0}" = 1 ] || [ ! -f "$NPZ"  ]; then uv run turbohead-build-clusters --head "$HEAD" --out "$NPZ" --cap "$CAP"; fi

# OPTIONAL per-model ScaNN-anisotropy (eta) sweep. OFF by default; the standard build ships eta=1.
# Set ETA_SWEEP="2 4" to enable: rebuild clusters at each eta, keep the one with the best real-splice
# top-1 agreement @P, and ONLY if it beats isotropic eta=1 (eta is per-model: it helps some models and
# regresses others, so the agreement gate is mandatory). Each extra eta costs one cluster rebuild
# (~7-20 min by vocab size) + ~1 min agreement. Picked eta recorded in $ROOT/eta.txt.
SWEEP=${ETA_SWEEP:-}
if [ -n "$SWEEP" ] && { [ "${FORCE:-0}" = 1 ] || [ ! -f "$ROOT/eta.txt" ]; }; then
  echo "== eta sweep: {1 $SWEEP}, real-splice agreement gate @P=$P =="
  agree_at() {  # $1=npz -> top-1 agreement % at P=$P (agreement reports a fixed grid; 256 if P off-grid)
    local gp=$P; case "$P" in 128|256|384|512) ;; *) gp=256;; esac
    uv run turbohead-agreement --npz "$1" --model "$MODEL" 2>&1 \
      | awk -v p="P=$gp " 'index($0,p){gsub(/%/,"",$NF); print $NF; exit}'
  }
  best_eta=1; best_ag=$(agree_at "$NPZ")
  echo "   eta=1  agree@$P = ${best_ag:-?}%"
  for e in $SWEEP; do
    cand="$ROOT/clusters_eta$e.npz"
    uv run turbohead-build-clusters --head "$HEAD" --out "$cand" --cap "$CAP" --eta "$e"
    ag=$(agree_at "$cand")
    echo "   eta=$e  agree@$P = ${ag:-?}%"
    if [ -n "$ag" ] && [ -n "$best_ag" ] && awk "BEGIN{exit !($ag > $best_ag)}"; then
      best_ag=$ag; best_eta=$e; cp "$cand" "$NPZ"   # promote: $NPZ now holds the best-so-far partition
    fi
    rm -f "$cand"
  done
  echo "$best_eta" > "$ROOT/eta.txt"
  echo "== eta sweep: kept eta=$best_eta (agree@$P=${best_ag:-?}%) =="
fi

# Calibrate the always-score frequent-miss list (~free top-1 agreement lift): the tokens FlashHead
# routes badly, always scored so they can't be missed. ALWAYS_SCORE = count to keep (default 64; 0 off).
ALWAYS=${ALWAYS_SCORE:-64}
ASARG=""
if [ "$ALWAYS" != 0 ]; then
  if [ "${FORCE:-0}" = 1 ] || [ ! -f "$ROOT/always_score.npy" ]; then
    uv run turbohead-calibrate-misses --model "$MODEL" --npz "$NPZ" --out "$ROOT/always_score.npy" -P "$P" --top "$ALWAYS"
  fi
  ASARG="--always-score $ROOT/always_score.npy"
fi

# dense-head baselines (the comparison points): fp32-eq, int8, int4 at two group sizes.
# SKIP_DENSE=1 omits them (CI smoke builds only the product = the flash splices below).
if [ "${SKIP_DENSE:-0}" != 1 ]; then
  uv run turbohead-quantize-head --src "$BASE" --head "$HEAD" --bits 16                 --dst "$ROOT/head16"
  uv run turbohead-quantize-head --src "$BASE" --head "$HEAD" --bits 8  --group-size 128 --dst "$ROOT/head8g128"
  uv run turbohead-quantize-head --src "$BASE" --head "$HEAD" --bits 4  --group-size 128 --dst "$ROOT/head4g128"
  uv run turbohead-quantize-head --src "$BASE" --head "$HEAD" --bits 4  --group-size 32  --dst "$ROOT/head4g32"
fi

# flash heads: portable onnx (logits-out) + fused custom op (shortlist-out). $ASARG adds always-score.
uv run turbohead-splice --backend onnx  --src "$BASE" --npz "$NPZ" --head "$HEAD" -P "$P" $ASARG --dst "$ROOT/onnx"
uv run turbohead-splice --backend fused --src "$BASE" --npz "$NPZ" --head "$HEAD" -P "$P" $ASARG --dst "$ROOT/fused"

cat <<EOF

== built (build log: $LOGS/build.log). now bench + eval ==
  HEADS="$ROOT/head16 $ROOT/head8g128 $ROOT/head4g128 $ROOT/head4g32 $ROOT/onnx $ROOT/fused"
  uv run turbohead-bench \$HEADS --threads 1,2,4,8 --reps 7                          2>&1 | tee $LOGS/bench_greedy.log
  uv run turbohead-bench \$HEADS --threads 1,2,4,8 --reps 7 --temperature 0.8 --seed 0 2>&1 | tee $LOGS/bench_sample.log
  uv run turbohead-head-quality --src $ROOT --npz $NPZ --head $HEAD -P $P            2>&1 | tee $LOGS/quality.log
EOF
