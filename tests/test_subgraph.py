"""Op-chain correctness on tiny synthetic clusters — no model artifact, runs in CI.

Gates the core of the method: that the spliced graph computes the *exact* dense logit
(h·W_row) for every token it scores. Probing all K clusters (P=K) makes coverage total,
so the flash output must reproduce the brute-force dense head bit-for-bit (stage 2 is fp32).

  - onnx backend (contract A): pure standard ops, always runs.
  - fused backend (contract H): the custom op; skipped unless csrc/libturbohead.so is built.
"""
import os
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, TensorProto

from turbohead.surgery.build_clusters import balanced_assign, build
from turbohead.surgery.build_subgraph import make_flash_nodes, fused_stage2_nodes

LIB = "csrc/libturbohead.so"


def _clusters(V=24, D=8, cap=4, seed=0):
    """Tiny balanced-k-means assets + the dense head [V,D] they realize. `build` stores Wperm
    in fp16, so the reference is reconstructed from Wperm/Vmap (not the original fp32 W) — that
    is the exact weight the spliced graph dots against, making the fp32 stage-2 comparison tight."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((V, D)).astype(np.float32)
    K = V // cap
    C = W[rng.choice(V, K, replace=False)].copy()
    Cnorm, Wperm, Vmap = build(W, balanced_assign(W, C, cap), cap)
    Wref = np.empty((V, D), np.float32)
    Wref[Vmap.reshape(-1)] = Wperm.reshape(V, D).astype(np.float32)
    return Wref, Cnorm, Wperm, Vmap, K, cap, D


def _session(model, so=None):
    return ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])


def test_onnx_chain_matches_dense_head_when_probing_all():
    """Contract A: P=K probes every cluster -> full (1,V) logits must equal the dense head."""
    Wref, Cnorm, Wperm, Vmap, K, cap, D = _clusters()
    V = K * cap
    special_ids = np.array([0, 5], np.int64)
    nodes, inits = make_flash_nodes(Cnorm, Wperm, Vmap, Wref[special_ids], special_ids,
                                    P=K, hidden="h", logits_out="logits", stage1="fp16")
    g = helper.make_graph(nodes, "flash",
                          [helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, D])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, V])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)   # pure standard ops -> must validate

    rng = np.random.default_rng(1)
    for _ in range(5):
        h = rng.standard_normal((1, D)).astype(np.float32)
        got = _session(m).run(["logits"], {"h": h})[0][0]
        np.testing.assert_allclose(got, (h @ Wref.T)[0], rtol=0, atol=1e-4)


@pytest.mark.skipif(not os.path.exists(LIB), reason=f"{LIB} not built (bash csrc/build.sh)")
@pytest.mark.parametrize("weight_dtype, atol", [("fp32", 1e-4), ("int8", 5e-2)])
def test_fused_op_matches_dense_head(weight_dtype, atol):
    """Contract H: the custom op's shortlist logits/ids must equal the dense reference for every
    scored token. int8 (FlashHeadSelectQ8) is per-row quantized -> looser tol, exact ids."""
    Wref, Cnorm, Wperm, Vmap, K, cap, D = _clusters()
    special_ids = np.array([0, 5], np.int64)
    nodes, inits = fused_stage2_nodes(Wperm, Vmap, Wref[special_ids], special_ids,
                                      hidden="h", ti1="ti1", weight_dtype=weight_dtype)
    g = helper.make_graph(
        nodes, "fused",
        [helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, D]),
         helper.make_tensor_value_info("ti1", TensorProto.INT64, [K])],
        [helper.make_tensor_value_info("cand_logits", TensorProto.FLOAT, [1, K * cap + 2]),
         helper.make_tensor_value_info("cand_ids", TensorProto.INT64, [1, K * cap + 2])],
        inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17),
                                            helper.make_opsetid("turbohead", 1)])
    so = ort.SessionOptions()
    so.register_custom_ops_library(LIB)
    sess = _session(m, so)

    rng = np.random.default_rng(2)
    ti = np.arange(K, dtype=np.int64)   # probe every cluster -> shortlist covers all V + specials
    for _ in range(5):
        h = rng.standard_normal((1, D)).astype(np.float32)
        lg, ids = (a[0] for a in sess.run(["cand_logits", "cand_ids"], {"h": h, "ti1": ti}))
        ref = (h @ Wref.T)[0]                   # dense logit per token id
        np.testing.assert_allclose(lg, ref[ids], rtol=0, atol=atol)
        assert sorted(ids.tolist()) == sorted(Vmap.reshape(-1).tolist() + special_ids.tolist())
