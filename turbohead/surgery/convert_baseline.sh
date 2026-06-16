#!/usr/bin/env bash
# §0.2 — build an HF model to ONNX: INT4 body, INT8 lm-head/embed, accuracy_level=4, group 128.
# Run from repo root: MODEL=Qwen/Qwen3-0.6B bash turbohead/surgery/convert_baseline.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
OUT=${OUT:-./artifacts/qwen3_0_6b_int4_cpu}
CACHE=${CACHE:-./artifacts/hf_cache}
# k_quant_last is the better int4 body calibration but its quantizer reshapes break on some weight
# shapes (e.g. Gemma3-270m). Set INT4_ALGO= (empty) for plain RTN, robust across shapes.
INT4_ALGO=${INT4_ALGO-k_quant_last}
ALGO_OPT=""; [ -n "$INT4_ALGO" ] && ALGO_OPT="int4_algo_config=$INT4_ALGO"

# _genai_build.py = builder + a Gemma3 rope-config compat shim (no-op for other models).
uv run python turbohead/surgery/_genai_build.py \
  -m "$MODEL" \
  -o "$OUT" \
  -p int4 \
  -e cpu \
  -c "$CACHE" \
  --extra_options int4_block_size=128 int4_accuracy_level=4 $ALGO_OPT hf_token=false
