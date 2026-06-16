#!/usr/bin/env bash
# Build the TurboHead fused custom-op kernel -> csrc/libturbohead.so.
# Fetches the matching ORT C/C++ headers (the pip wheel ships none) then compiles.
# Header-only against the ORT API; no libonnxruntime link needed.
# Run from repo root: bash csrc/build.sh
set -euo pipefail
cd "$(dirname "$0")"

VER=${ORT_VERSION:-1.26.0}
INC=include
mkdir -p "$INC"
BASE="https://raw.githubusercontent.com/microsoft/onnxruntime/v${VER}/include/onnxruntime/core/session"
# seed headers; resolve transitive #include "..." in a couple of passes
for h in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h \
         onnxruntime_lite_custom_op.h onnxruntime_float16.h onnxruntime_ep_c_api.h; do
  [ -f "$INC/$h" ] || curl -fsSL "$BASE/$h" -o "$INC/$h"
done
for _ in 1 2 3; do
  miss=$(grep -rhoE '#include "[a-z_]+\.h"' "$INC"/*.h | sed -E 's/#include "([^"]+)"/\1/' | sort -u \
         | while read -r h; do [ -f "$INC/$h" ] || echo "$h"; done)
  [ -z "$miss" ] && break
  for h in $miss; do curl -fsSL "$BASE/$h" -o "$INC/$h"; done
done

# -ffast-math: lets the dot-product reduction auto-vectorize (strict FP ordering
#   otherwise forces a scalar loop ~2x slower). Kernel is serial by design — see the
#   note in turbohead_op.cc (OpenMP tested, no gain: the loop is memory-bound).
g++ -O3 -march=native -ffast-math -funroll-loops -std=c++17 -shared -fPIC \
    -I"$INC" turbohead_op.cc -o libturbohead.so
echo "built csrc/libturbohead.so"
