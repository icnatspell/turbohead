// TurboHead fused stage-2. Replaces the spliced fh_ chain (Gather Wperm -> MatMul
// -> Concat -> ScatterElements) with one pass: for each probed cluster, dot its cap
// rows with h. Emits the *candidate shortlist* (logits + token ids) for the ~N
// scored tokens — Python then does greedy (argmax) or sampling (softmax over the
// shortlist), skipping the full (1,V) logits, the scatter, and the full-vocab softmax.
// Reads each candidate weight row exactly once; no (P*cap,D) materialization.
//
// Build: bash csrc/build.sh   (needs -fopenmp + -ffast-math; see that script)
// Header-only against the ORT C/C++ API; no libonnxruntime link needed.
// ponytail: include std headers before ORT to avoid GCC/aarch64 namespace-nesting bug
#include <optional>
#include <numeric>
#include <unordered_set>
#include <cmath>
#include <memory>
#define ORT_API_MANUAL_INIT  // custom-op lib: set the global API ptr ourselves in RegisterCustomOps
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"

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

  // Serial on purpose: this loop is memory-bound (streams ~P*cap*D*4 = 16.8MB of weight
  // rows), so one thread already saturates the bandwidth that matters. Tested OpenMP over
  // p (disjoint writes, no reduction) — no gain at 4 threads, ~7% SLOWER at 8 (fork/join +
  // oversubscription vs ORT's spinning idle threads). The body's ORT threads parallelize
  // the heavy matmuls; this ~2ms head is not the bottleneck.
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

// int8 variant: Wperm rows are symmetric per-row int8 (scale[K,cap] dequants the dot).
// Specials stay fp32 (only S rows, not worth quantizing). Same loop, ~4x less weight
// traffic — see docs/FUSED_HEAD_INT8.md. Quant is per output-channel so logits are accurate
// well above the body's 4-bit weights.
void FlashHeadSelectQ8(const Tensor<float>&   h,
                       const Tensor<int64_t>& ti,
                       const Tensor<int8_t>&  Wperm,
                       const Tensor<float>&   scale,
                       const Tensor<int64_t>& Vmap,
                       const Tensor<float>&   Wspec,
                       const Tensor<int64_t>& spec_ids,
                       Tensor<float>&         cand_logits,
                       Tensor<int64_t>&       cand_ids) {
  const float*   hp  = h.Data();
  const int64_t* tip = ti.Data();
  const int8_t*  W   = Wperm.Data();
  const float*   sc  = scale.Data();
  const int64_t* V   = Vmap.Data();
  const auto&    ws  = Wperm.Shape();          // [K, cap, D]
  const int64_t  cap = ws[1], D = ws[2];
  const int64_t  P   = ti.Shape()[0];
  const int64_t  S   = spec_ids.Shape()[0];
  const int64_t  N   = P * cap + S;

  float*   lg = cand_logits.Allocate({1, N});
  int64_t* id = cand_ids.Allocate({1, N});

  for (int64_t p = 0; p < P; ++p) {
    const int64_t  c    = tip[p];
    const int8_t*  rows = W + c * cap * D;
    const int64_t* vid  = V + c * cap;
    const float*   rsc  = sc + c * cap;
    for (int64_t r = 0; r < cap; ++r) {
      const int8_t* row = rows + r * D;
      float acc = 0.f;
      for (int64_t d = 0; d < D; ++d) acc += (float)row[d] * hp[d];  // int8->float, scale hoisted out
      const int64_t idx = p * cap + r;
      lg[idx] = acc * rsc[r];
      id[idx] = vid[r];
    }
  }
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
  static std::unique_ptr<Ort::Custom::OrtLiteCustomOp> op8{
      Ort::Custom::CreateLiteCustomOp("FlashHeadSelectQ8", "CPUExecutionProvider", FlashHeadSelectQ8)};
  OrtCustomOpDomain* domain = nullptr;
  if (auto* st = api->CreateCustomOpDomain("turbohead", &domain)) return st;
  if (auto* st = api->CustomOpDomain_Add(domain, op.get())) return st;
  if (auto* st = api->CustomOpDomain_Add(domain, op8.get())) return st;
  return api->AddCustomOpDomain(options, domain);
}
 