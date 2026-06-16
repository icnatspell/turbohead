#!/usr/bin/env bash
# §0.2 — build an HF model to ONNX: INT4 body, INT8 lm-head/embed, accuracy_level=4, group 128.
# Run from repo root: MODEL=Qwen/Qwen3-0.6B bash turbohead/surgery/convert_baseline.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
OUT=${OUT:-./artifacts/qwen3_0_6b_int4_cpu}
CACHE=${CACHE:-./artifacts/hf_cache}
# Default to plain RTN int4 for the body: robust across all shapes/sizes, whereas k_quant_last's
# quantizer reshape-crashes on some models (Gemma3) and risks the 2GB protobuf limit on others
# (Llama). The body algo is irrelevant to our head comparison — decode speed is identical (same
# MatMulNBits op) and head-quality is measured on fp32 hidden states. Set INT4_ALGO=k_quant_last
# to opt back in for a model where it works.
INT4_ALGO=${INT4_ALGO-}
ALGO_OPT=""; [ -n "$INT4_ALGO" ] && ALGO_OPT="int4_algo_config=$INT4_ALGO"

# _genai_build.py = builder + a Gemma3 rope-config compat shim (no-op for other models).
uv run python turbohead/surgery/_genai_build.py \
  -m "$MODEL" \
  -o "$OUT" \
  -p int4 \
  -e cpu \
  -c "$CACHE" \
  --extra_options int4_block_size=128 int4_accuracy_level=4 $ALGO_OPT hf_token=false
