#!/usr/bin/env bash
# §0.2 — build Qwen3-0.6B to ONNX: INT4 body, INT8 lm-head/embed, accuracy_level=4, group 128.
# Run from repo root: bash turbohead/surgery/convert_baseline.sh
set -euo pipefail

OUT=${OUT:-./artifacts/qwen3_0_6b_int4_cpu}
CACHE=${CACHE:-./artifacts/hf_cache}

uv run python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen3-0.6B \
  -o "$OUT" \
  -p int4 \
  -e cpu \
  -c "$CACHE" \
  --extra_options int4_block_size=128 int4_accuracy_level=4 int4_algo_config=k_quant_last hf_token=false
