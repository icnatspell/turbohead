"""FlashHead subgraph builder — the portable onnx stage-2 chain that emits full (1,V) logits.
make_flash_nodes(): nodes+inits to splice into a real model (hidden -> logits name).
Standalone __main__: wrap with an `h` input, verify argmax == dense on real hidden states.

Op chain (fp32 accumulation):
  sims   = h @ Cnorm                      (1,K)
  top    = TopK(sims,P).indices           (1,P)
  rows   = Gather(Wperm,top) -> (P*cap,D)
  l2     = MatMul(rows, hT)               (P*cap,1)  probed-token logits
  ls     = MatMul(Wspec, hT)              (S,1)      always-scored specials (EOS)
  ids    = concat(Gather(Vmap,top).flat, special_ids)
  logits = ScatterElements(-inf base, ids, concat(l2,ls), axis=1)   (1,V)
"""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from loguru import logger

NEG = -1e9  # forced-negative fill for unscored vocab


def _f16(name, arr):
    return numpy_helper.from_array(arr.astype(np.float16), name)


def _i64(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.int64), name)


def _per_row_int8(Wperm):
    """Symmetric per-output-row (== per-output-channel) int8 of Wperm (K,cap,D).
    Returns (q int8 (K,cap,D), scale fp32 (K,cap)). Shared by the fused op and the onnx int8 path."""
    K, cap, D = Wperm.shape
    rows = Wperm.reshape(K * cap, D).astype(np.float32)
    scale = (np.maximum(np.abs(rows).max(axis=1), 1e-9) / 127.0).astype(np.float32)
    q = np.clip(np.rint(rows / scale[:, None]), -127, 127).astype(np.int8)
    return q.reshape(K, cap, D), scale.reshape(K, cap)


STAGE1_CHOICES = ("fp16", "int8", "int8ch", "int4")


def _int8ch_sims_nodes(Cnorm, hidden, n):
    """Per-CHANNEL (per-centroid) int8 stage-1 sims via MatMulInteger — the canonical ONNX op for
    per-channel weights, since MatMulNBits is block-wise (and caps block at 256). h is dynamically
    int8-quantized; Cnorm is symmetric per-column int8. The per-channel weight scale is a `Mul`
    *outside* the integer matmul (MatMulInteger has no fused dequant, unlike MatMulNBits — so this is
    correct-but-slower at M=1; per-group int8 dominates it on speed, see docs/ORT_QUIRKS.md).
        sims[k] = h_scale * wscale[k] * MatMulInteger(hq - h_zp, Cq)[k]
    """
    wscale = (np.maximum(np.abs(Cnorm).max(axis=0), 1e-9) / 127.0).astype(np.float32)  # (K,) per-col
    Cq = np.clip(np.rint(Cnorm / wscale), -127, 127).astype(np.int8)                    # (D,K)
    nodes = [
        helper.make_node("DynamicQuantizeLinear", [hidden], [n("hq"), n("hsc"), n("hzp")]),  # uint8
        helper.make_node("MatMulInteger", [n("hq"), n("Cq"), n("hzp")], [n("mm_i32")]),      # (1,K) int32
        helper.make_node("Cast", [n("mm_i32")], [n("mm_f32")], to=TensorProto.FLOAT),
        helper.make_node("Mul", [n("mm_f32"), n("wscale")], [n("mm_ws")]),                   # * per-ch scale
        helper.make_node("Mul", [n("mm_ws"), n("hsc")], [n("sims")]),                        # * act scale
    ]
    inits = [numpy_helper.from_array(Cq, n("Cq")), numpy_helper.from_array(wscale, n("wscale"))]
    return nodes, inits


def stage1_nodes(Cnorm, P, hidden, pfx="fh_", stage1="int4"):
    """Stage 1 (shared by both backends) — score h against the K centroids, keep the top-P
    clusters. Returns (nodes, inits, ti1) where ti1 (P,) names the probed cluster indices.

      "fp16"   — Cast h -> fp16 MatMul.
      "int8"   — per-GROUP int8. fp32 MatMul named `<pfx>sims_mm`, rewritten to MatMulNBits (W8A16)
                 by quantize_stage1() post-build. ORT's fused-dequant gemv beats fp16 at M=1.
      "int4"   — per-GROUP int4, same path (W4A16, ~9x at M=1; the paper's stage-1 trick).
      "int8ch" — per-CHANNEL int8 via MatMulInteger, built inline here (no post-build rewrite).
    """
    def n(s):
        return pfx + s
    if stage1 in ("int4", "int8"):
        s1 = [helper.make_node("MatMul", [hidden, n("Cnorm")], [n("sims")], name=n("sims_mm"))]
        s1_inits = [numpy_helper.from_array(Cnorm.astype(np.float32), n("Cnorm"))]
    elif stage1 == "int8ch":
        s1, s1_inits = _int8ch_sims_nodes(Cnorm, hidden, n)
    elif stage1 == "fp16":
        s1 = [helper.make_node("Cast", [hidden], [n("h16")], to=TensorProto.FLOAT16),
              helper.make_node("MatMul", [n("h16"), n("Cnorm")], [n("sims")], name=n("sims_mm"))]
        s1_inits = [_f16(n("Cnorm"), Cnorm)]
    else:
        raise ValueError(f"stage1 must be one of {STAGE1_CHOICES}, got {stage1!r}")
    nodes = s1 + [
        helper.make_node("TopK", [n("sims"), n("Pk")], [n("tv"), n("ti")], axis=1, sorted=0),
        helper.make_node("Squeeze", [n("ti"), n("ax0")], [n("ti1")]),
    ]
    inits = s1_inits + [_i64(n("Pk"), [P]), _i64(n("ax0"), [0])]
    return nodes, inits, n("ti1")


def onnx_stage2_nodes(Wperm, Vmap, Wspec, special_ids, P, hidden, ti1, logits_out, pfx="fh_",
                      weight_dtype="fp32"):
    """Stage 2, **logits-out** (portable, no custom op): gather the P*cap candidate rows, dot with
    h, scatter into a (1,V) -1e9 base -> full logits at `logits_out`.

    weight_dtype:
      'fp32' — store the candidate rows fp32. The CPU EP has no fp16 matmul kernel (it inserts a
               fp32 cast that costs more than the bandwidth it saves — docs/ORT_QUIRKS.md), so fp32
               is the faster choice for a stock-ops graph.
      'int8' — store Wperm per-output-row (== per-channel) symmetric int8 + a (K,cap) scale. Gather
               the int8 rows, Cast->fp32, dot, then Mul the (P*cap,1) result by the gathered per-row
               scale (scale hoisted out of the dot, mirroring the fused FlashHeadSelectQ8). 4x less
               weight read at the Gather, BUT the Cast re-materializes the fp32 rows so the matmul-
               input bandwidth is unchanged — expect flat-to-negative on CPU EP (docs/ORT_QUIRKS.md).
               Specials stay fp32.
    """
    K, cap, D = Wperm.shape
    V, N = K * cap, P * cap + len(special_ids)
    def n(s):
        return pfx + s
    hT = helper.make_node("Transpose", [hidden], [n("hT")], perm=[1, 0])           # (D,1)
    if weight_dtype == "int8":
        q, scale = _per_row_int8(Wperm.reshape(K, cap, D))
        l2_nodes = [
            helper.make_node("Gather", [n("Wperm_q8"), ti1], [n("wg_q8")], axis=0),    # (P,cap,D) int8
            helper.make_node("Cast", [n("wg_q8")], [n("wg")], to=TensorProto.FLOAT),
            helper.make_node("Reshape", [n("wg"), n("shp_pcD")], [n("rows")]),         # (P*cap,D)
            hT,
            helper.make_node("MatMul", [n("rows"), n("hT")], [n("l2_raw")]),           # (P*cap,1)
            helper.make_node("Gather", [n("Wperm_scale"), ti1], [n("sg")], axis=0),    # (P,cap)
            helper.make_node("Reshape", [n("sg"), n("shp_pc1")], [n("sflat")]),        # (P*cap,1)
            helper.make_node("Mul", [n("l2_raw"), n("sflat")], [n("l2")]),             # scale hoisted out
        ]
        l2_inits = [numpy_helper.from_array(q, n("Wperm_q8")),
                    numpy_helper.from_array(scale, n("Wperm_scale")),
                    _i64(n("shp_pc1"), [P * cap, 1])]
    elif weight_dtype == "fp32":
        l2_nodes = [
            helper.make_node("Gather", [n("WpermF32"), ti1], [n("wg")], axis=0),       # (P,cap,D) fp32
            helper.make_node("Reshape", [n("wg"), n("shp_pcD")], [n("rows")]),         # (P*cap,D)
            hT,
            helper.make_node("MatMul", [n("rows"), n("hT")], [n("l2")]),               # (P*cap,1)
        ]
        l2_inits = [numpy_helper.from_array(Wperm.reshape(K, cap, D).astype(np.float32), n("WpermF32"))]
    else:
        raise ValueError(f"weight_dtype must be 'fp32' or 'int8', got {weight_dtype!r}")
    nodes = l2_nodes + [
        helper.make_node("MatMul", [n("WspecF32"), n("hT")], [n("ls")]),           # (S,1) specials
        helper.make_node("Concat", [n("l2"), n("ls")], [n("updates2")], axis=0),   # (N,1)
        helper.make_node("Reshape", [n("updates2"), n("shp_1N")], [n("updates")]), # (1,N)
        helper.make_node("Gather", [n("Vmap"), ti1], [n("vg")], axis=0),           # candidate ids
        helper.make_node("Reshape", [n("vg"), n("shp_pc")], [n("vflat")]),
        helper.make_node("Concat", [n("vflat"), n("spec_ids")], [n("ids")], axis=0),
        helper.make_node("Reshape", [n("ids"), n("shp_1N")], [n("idx")]),
        helper.make_node("ScatterElements", [n("base"), n("idx"), n("updates")], [logits_out], axis=1),
    ]
    inits = l2_inits + [
        numpy_helper.from_array(Wspec.astype(np.float32), n("WspecF32")),
        _i64(n("Vmap"), Vmap),
        _i64(n("spec_ids"), special_ids),
        _i64(n("shp_pcD"), [P * cap, D]),
        _i64(n("shp_pc"), [P * cap]),
        _i64(n("shp_1N"), [1, N]),
        numpy_helper.from_array(np.full((1, V), NEG, np.float32), n("base")),
    ]
    return nodes, inits


def fused_stage2_nodes(Wperm, Vmap, Wspec, special_ids, hidden, ti1,
                       logits_out="cand_logits", ids_out="cand_ids", pfx="fh_",
                       weight_dtype="fp32"):
    """Stage 2, **shortlist-out** (custom op): one op reads only the probed rows and emits the
    candidate (logits, ids) shortlist of length N = P*cap + S. No (P*cap,D) materialization,
    no (1,V), no scatter. Needs csrc/libturbohead.so registered at inference.

    weight_dtype: 'fp32' -> FlashHeadSelect (16.8MB/token read). 'int8' -> FlashHeadSelectQ8:
    Wperm stored per-output-channel int8 (4x less weight traffic; the head is memory-bound, so
    this directly speeds the serial head — see docs/FUSED_HEAD_INT8.md). Specials stay fp32."""
    K, cap, D = Wperm.shape
    def n(s):
        return pfx + s
    common = [
        _i64(n("Vmap"), Vmap),
        numpy_helper.from_array(Wspec.astype(np.float32), n("Wspec")),
        _i64(n("spec_ids"), special_ids),
    ]
    if weight_dtype == "int8":
        q, scale = _per_row_int8(Wperm.reshape(K, cap, D))
        nodes = [helper.make_node(
            "FlashHeadSelectQ8",
            [hidden, ti1, n("Wperm_q8"), n("Wperm_scale"), n("Vmap"), n("Wspec"), n("spec_ids")],
            [logits_out, ids_out], domain="turbohead")]
        inits = [numpy_helper.from_array(q, n("Wperm_q8")),
                 numpy_helper.from_array(scale, n("Wperm_scale"))] + common
        return nodes, inits
    if weight_dtype != "fp32":
        raise ValueError(f"weight_dtype must be 'fp32' or 'int8', got {weight_dtype!r}")
    nodes = [helper.make_node(
        "FlashHeadSelect",
        [hidden, ti1, n("Wperm"), n("Vmap"), n("Wspec"), n("spec_ids")],
        [logits_out, ids_out], domain="turbohead")]
    inits = [numpy_helper.from_array(Wperm.reshape(K, cap, D).astype(np.float32), n("Wperm"))] + common
    return nodes, inits


def make_flash_nodes(Cnorm, Wperm, Vmap, Wspec, special_ids, P, hidden, logits_out, pfx="fh_",
                     stage1="fp16", weight_dtype="fp32"):
    """logits-out flash head as one block (stage1 + onnx stage2): `hidden` (1,D) -> (1,V) at
    `logits_out`. Thin wrapper over stage1_nodes + onnx_stage2_nodes; kept for the standalone
    gate (build_standalone) and any caller wanting the whole subgraph in one call."""
    s1n, s1i, ti1 = stage1_nodes(Cnorm, P, hidden, pfx, stage1)
    s2n, s2i = onnx_stage2_nodes(Wperm, Vmap, Wspec, special_ids, P, hidden, ti1, logits_out, pfx,
                                 weight_dtype)
    return s1n + s2n, s1i + s2i


# MLAS' MatMulNBits CPU kernel silently computes garbage above this block size (verified: correct
# at 128/256, argmax 0% at 512/1024, both accuracy_levels — docs/ORT_QUIRKS.md). So per-GROUP int8/int4
# can't go coarser than 256; per-CHANNEL int8 uses MatMulInteger instead (stage1="int8ch").
MATMULNBITS_MAX_BLOCK = 256


def quantize_stage1(model, stage1, block_size=128, acc_level=4, pfx="fh_"):
    """Rewrite only the stage-1 MatMul (`<pfx>sims_mm`) to MatMulNBits (W4A16 for int4, W8A16
    for int8). The dense body is already MatMulNBits; stage-2 stays fp32. `block_size` is the
    quant group size along the contraction dim (must divide D; e.g. 32/64/128/256). Returns ModelProto."""
    from onnxruntime.quantization import matmul_nbits_quantizer as mnq
    if block_size > MATMULNBITS_MAX_BLOCK:
        raise ValueError(
            f"stage-1 block_size={block_size} > {MATMULNBITS_MAX_BLOCK}: MLAS' MatMulNBits CPU kernel "
            f"silently computes garbage above {MATMULNBITS_MAX_BLOCK} (see docs/ORT_QUIRKS.md). "
            f"Use <= {MATMULNBITS_MAX_BLOCK} for per-group, or stage1='int8ch' for true per-channel.")
    bits = {"int4": 4, "int8": 8}[stage1]
    q = mnq.MatMulNBitsQuantizer(model, bits=bits, block_size=block_size, is_symmetric=True,
                                 accuracy_level=acc_level, nodes_to_include=[pfx + "sims_mm"])
    q.process()
    return q.model.model


def build_standalone(npz_path, P, special_ids, out="flash_standalone.onnx", stage1="int4",
                     block_size=128, weight_dtype="fp32"):
    z = np.load(npz_path)
    Cnorm, Wperm, Vmap = z["Cnorm"], z["Wperm"], z["Vmap"]
    K, cap, D = Wperm.shape
    V = K * cap
    W = np.load("artifacts/head_W.npy")  # for special rows
    Wspec = W[special_ids].astype(np.float16)
    nodes, inits = make_flash_nodes(Cnorm, Wperm, Vmap, Wspec, special_ids, P, "h", "logits",
                                    stage1=stage1, weight_dtype=weight_dtype)
    h = helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, D])
    out_vi = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, V])
    g = helper.make_graph(nodes, "flash", [h], [out_vi], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    if stage1 in ("int4", "int8"):
        m = quantize_stage1(m, stage1, block_size)  # fh_sims_mm -> MatMulNBits (com.microsoft)
    else:
        onnx.checker.check_model(m)  # checker rejects com.microsoft contrib ops
    for f in (out, out + ".data"):  # external_data save errors if stale file exists
        Path(f).unlink(missing_ok=True)
    onnx.save(m, out, save_as_external_data=True, location=out + ".data")
    logger.info(f"saved {out}  (P={P}, V={V}, D={D}, stage1={stage1}, block_size={block_size})")
    return out


def main():
    # Phase 2 gate: standalone subgraph argmax == dense head argmax on real hidden states.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import onnxruntime as ort

    ap = argparse.ArgumentParser(description="Standalone subgraph argmax vs dense head (fp16/int8/int4 sweep)")
    ap.add_argument("-P", "--probes", type=int, default=256)
    a = ap.parse_args()
    P = a.probes
    special_ids = np.array([151643, 151645], np.int64)  # bos, eos (Qwen3 ids; standalone verifier only)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
    h_in = {}
    hook = model.lm_head.register_forward_hook(lambda m, i, o: h_in.update(h=i[0]))
    ids = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids
    with torch.no_grad():
        dense = model(ids).logits[0].argmax(-1).numpy()
    hook.remove()
    H = h_in["h"][0].float().numpy()

    def agree_pct(path):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        a = sum(int(sess.run(["logits"], {"h": H[i : i + 1].astype(np.float32)})[0].argmax() == dense[i])
                for i in range(H.shape[0]))
        return a, H.shape[0]

    # A/B stage-1 precision: does quantized centroid scoring hold accuracy vs fp16?
    # int8 = per-group (MatMulNBits); int8ch = per-channel (MatMulInteger).
    for stage1 in ("fp16", "int8", "int8ch", "int4"):
        a, t = agree_pct(build_standalone("artifacts/clusters.npz", P, special_ids, stage1=stage1,
                                          out=f"flash_s1_{stage1}.onnx"))
        logger.info(f"stage1={stage1:6s}  flash vs dense argmax: {a}/{t} = {a / t:.1%}")
    # A/B stage-2 candidate-row precision (stage1=int4 fixed): per-channel int8 rows vs fp32.
    for wd in ("fp32", "int8"):
        a, t = agree_pct(build_standalone("artifacts/clusters.npz", P, special_ids, stage1="int4",
                                          weight_dtype=wd, out=f"flash_s2_{wd}.onnx"))
        logger.info(f"stage2={wd:4s}  flash vs dense argmax: {a}/{t} = {a / t:.1%}")


if __name__ == "__main__":
    main()
