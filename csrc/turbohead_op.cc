// TurboHead fused stage-2, greedy / contract-B. Replaces the spliced fh_ chain
// (Gather Wperm -> MatMul -> Concat -> ScatterElements -> argmax) with a single
// pass: for each probed cluster, dot its cap rows with h and track the argmax.
// Reads each candidate weight row exactly once (no (P*cap,D) materialization,
// no (1,V) logits), and emits the next-token id directly.
//
// Build: g++ -O3 -march=native -std=c++17 -shared -fPIC -Icsrc/include \
//            csrc/turbohead_op.cc -o csrc/libturbohead.so
// Header-only against the ORT C/C++ API; no libonnxruntime link needed.
#define ORT_API_MANUAL_INIT  // custom-op lib: set the global API ptr ourselves in RegisterCustomOps
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include <cmath>
#include <memory>

using Ort::Custom::Tensor;

namespace {
// inputs: h (1,D) | ti (P,) probed clusters | Wperm (K,cap,D) | Vmap (K,cap)
//         Wspec (S,D) always-scored rows (EOS/BOS) | spec_ids (S,)
// output: next_token (1,1) int64  -> contract B (decode loop reads .flat[-1])
void FlashHeadGreedy(const Tensor<float>&   h,
                     const Tensor<int64_t>& ti,
                     const Tensor<float>&   Wperm,
                     const Tensor<int64_t>& Vmap,
                     const Tensor<float>&   Wspec,
                     const Tensor<int64_t>& spec_ids,
                     Tensor<int64_t>&       out) {
  const float*   hp  = h.Data();
  const int64_t* tip = ti.Data();
  const float*   W   = Wperm.Data();
  const int64_t* V   = Vmap.Data();
  const auto&    ws  = Wperm.Shape();          // [K, cap, D]
  const int64_t  cap = ws[1], D = ws[2];
  const int64_t  P   = ti.Shape()[0];

  float   best = -INFINITY;
  int64_t tok  = -1;
  for (int64_t p = 0; p < P; ++p) {
    const int64_t  c    = tip[p];
    const float*   rows = W + c * cap * D;
    const int64_t* vid  = V + c * cap;
    for (int64_t r = 0; r < cap; ++r) {
      const float* row = rows + r * D;
      float dot = 0.f;
      for (int64_t d = 0; d < D; ++d) dot += row[d] * hp[d];   // auto-vectorized at -O3 -march=native
      if (dot > best) { best = dot; tok = vid[r]; }
    }
  }
  // always-score specials so greedy can emit EOS (matches contract-A graph)
  const float*   Ws  = Wspec.Data();
  const int64_t* sid = spec_ids.Data();
  const int64_t  S   = spec_ids.Shape()[0];
  for (int64_t s = 0; s < S; ++s) {
    const float* row = Ws + s * D;
    float dot = 0.f;
    for (int64_t d = 0; d < D; ++d) dot += row[d] * hp[d];
    if (dot > best) { best = dot; tok = sid[s]; }
  }
  out.Allocate({1, 1})[0] = tok;
}
}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                     const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  static std::unique_ptr<Ort::Custom::OrtLiteCustomOp> op{
      Ort::Custom::CreateLiteCustomOp("FlashHeadGreedy", "CPUExecutionProvider", FlashHeadGreedy)};
  OrtCustomOpDomain* domain = nullptr;
  if (auto* st = api->CreateCustomOpDomain("turbohead", &domain)) return st;
  if (auto* st = api->CustomOpDomain_Add(domain, op.get())) return st;
  return api->AddCustomOpDomain(options, domain);
}
