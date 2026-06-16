// TurboHead fused stage-2. Replaces the spliced fh_ chain (Gather Wperm -> MatMul
// -> Concat -> ScatterElements) with one pass: for each probed cluster, dot its cap
// rows with h. Emits the *candidate shortlist* (logits + token ids) for the ~N
// scored tokens — Python then does greedy (argmax) or sampling (softmax over the
// shortlist), skipping the full (1,V) logits, the scatter, and the full-vocab softmax.
// Reads each candidate weight row exactly once; no (P*cap,D) materialization.
//
// Build: bash csrc/build.sh   (needs -fopenmp + -ffast-math; see that script)
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
// outputs: cand_logits (1,N) | cand_ids (1,N)   where N = P*cap + S
//          probed candidates first (cluster-major), specials last.
void FlashHeadSelect(const Tensor<float>&   h,
                     const Tensor<int64_t>& ti,
                     const Tensor<float>&   Wperm,
                     const Tensor<int64_t>& Vmap,
                     const Tensor<float>&   Wspec,
                     const Tensor<int64_t>& spec_ids,
                     Tensor<float>&         cand_logits,
                     Tensor<int64_t>&       cand_ids) {
  const float*   hp  = h.Data();
  const int64_t* tip = ti.Data();
  const float*   W   = Wperm.Data();
  const int64_t* V   = Vmap.Data();
  const auto&    ws  = Wperm.Shape();          // [K, cap, D]
  const int64_t  cap = ws[1], D = ws[2];
  const int64_t  P   = ti.Shape()[0];
  const int64_t  S   = spec_ids.Shape()[0];
  const int64_t  N   = P * cap + S;

  float*   lg = cand_logits.Allocate({1, N});
  int64_t* id = cand_ids.Allocate({1, N});

  // Embarrassingly parallel: each p writes a disjoint [p*cap, p*cap+cap) slice.
  // No reduction (argmax moved to Python over the shortlist), so no races.
#pragma omp parallel for schedule(static)
  for (int64_t p = 0; p < P; ++p) {
    const int64_t  c    = tip[p];
    const float*   rows = W + c * cap * D;
    const int64_t* vid  = V + c * cap;
    for (int64_t r = 0; r < cap; ++r) {
      const float* row = rows + r * D;
      float dot = 0.f;
      for (int64_t d = 0; d < D; ++d) dot += row[d] * hp[d];  // -ffast-math -> auto-vectorized
      const int64_t idx = p * cap + r;
      lg[idx] = dot;
      id[idx] = vid[r];
    }
  }
  // always-score specials (EOS/BOS) so greedy/sampling can emit them
  const float*   Ws  = Wspec.Data();
  const int64_t* sid = spec_ids.Data();
  for (int64_t s = 0; s < S; ++s) {
    const float* row = Ws + s * D;
    float dot = 0.f;
    for (int64_t d = 0; d < D; ++d) dot += row[d] * hp[d];
    lg[P * cap + s] = dot;
    id[P * cap + s] = sid[s];
  }
}
}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                     const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  static std::unique_ptr<Ort::Custom::OrtLiteCustomOp> op{
      Ort::Custom::CreateLiteCustomOp("FlashHeadSelect", "CPUExecutionProvider", FlashHeadSelect)};
  OrtCustomOpDomain* domain = nullptr;
  if (auto* st = api->CreateCustomOpDomain("turbohead", &domain)) return st;
  if (auto* st = api->CustomOpDomain_Add(domain, op.get())) return st;
  return api->AddCustomOpDomain(options, domain);
}
