"""Phase 2/3 — FlashHead subgraph, contract A (logits-shaped (1,V)). §2,§5,§6.
make_flash_nodes(): nodes+inits to splice into a real model (hidden -> logits name).
Standalone __main__: wrap with an `h` input, verify argmax == dense on real hidden states.

Op chain (fp32 accumulation, §6.4):
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


def make_flash_nodes(Cnorm, Wperm, Vmap, Wspec, special_ids, P, hidden, logits_out, pfx="fh_",
                     stage1="fp16"):
    """Return (nodes, initializers). `hidden` is a (1,D) tensor name; emits (1,V) at `logits_out`.

    stage1 (centroid-scoring precision):
      "fp16"        — Cast h -> fp16 MatMul.
      "int8"/"int4" — fp32 MatMul on the named node `<pfx>sims_mm`, rewritten to MatMulNBits
                      (W4A16 / W8A16) post-build by quantize_stage1(). ORT's fused-dequant gemv
                      is far faster than fp16 at M=1 (int4 ~9x); the paper's stage-1 trick.
    """
    K, cap, D = Wperm.shape
    V = K * cap
    S = len(special_ids)
    N = P * cap + S
    def n(s):
        return pfx + s

    # Stage 1 (centroid scoring). fp16 cast buys ~nothing (fp16 1.47ms ~= fp32 1.48ms at M=1);
    # quantized leaves it as a named fp32 MatMul so MatMulNBitsQuantizer can rewrite just this node.
    if stage1 in ("int4", "int8"):
        stage1_nodes = [helper.make_node("MatMul", [hidden, n("Cnorm")], [n("sims")],
                                         name=n("sims_mm"))]
        cnorm_init = numpy_helper.from_array(Cnorm.astype(np.float32), n("Cnorm"))
    elif stage1 == "fp16":
        stage1_nodes = [
            helper.make_node("Cast", [hidden], [n("h16")], to=TensorProto.FLOAT16),
            helper.make_node("MatMul", [n("h16"), n("Cnorm")], [n("sims")], name=n("sims_mm")),
        ]
        cnorm_init = _f16(n("Cnorm"), Cnorm)  # fp16 stage1, ~19MB/step bandwidth
    else:
        raise ValueError(f"stage1 must be fp16|int8|int4, got {stage1!r}")

    nodes = [
        *stage1_nodes,
        helper.make_node("TopK", [n("sims"), n("Pk")], [n("tv"), n("ti")], axis=1, sorted=0),
        helper.make_node("Squeeze", [n("ti"), n("ax0")], [n("ti1")]),
        # Stage2 fp32: CPU EP has no fp16 matmul kernel (it auto-casts), so store fp32 and skip
        # the implicit fp16->fp32 cast of the gathered rows (~2ms). Costs 2x gather bandwidth.
        helper.make_node("Gather", [n("WpermF32"), n("ti1")], [n("wg")], axis=0),  # (P,cap,D) fp32
        helper.make_node("Reshape", [n("wg"), n("shp_pcD")], [n("rows")]),         # (P*cap,D) fp32
        helper.make_node("Transpose", [hidden], [n("hT")], perm=[1, 0]),           # (D,1) fp32
        helper.make_node("MatMul", [n("rows"), n("hT")], [n("l2")]),               # (P*cap,1) fp32
        helper.make_node("MatMul", [n("WspecF32"), n("hT")], [n("ls")]),           # (S,1) fp32
        helper.make_node("Concat", [n("l2"), n("ls")], [n("updates2")], axis=0),   # (N,1) fp32
        helper.make_node("Reshape", [n("updates2"), n("shp_1N")], [n("updates")]), # (1,N) fp32
        # ids: Gather(Vmap,top) -> (P,cap) -> flat (P*cap,), concat specials -> (N,) -> (1,N)
        helper.make_node("Gather", [n("Vmap"), n("ti1")], [n("vg")], axis=0),
        helper.make_node("Reshape", [n("vg"), n("shp_pc")], [n("vflat")]),
        helper.make_node("Concat", [n("vflat"), n("spec_ids")], [n("ids")], axis=0),
        helper.make_node("Reshape", [n("ids"), n("shp_1N")], [n("idx")]),
        helper.make_node("ScatterElements", [n("base"), n("idx"), n("updates")], [logits_out], axis=1),
    ]
    inits = [
        cnorm_init,
        numpy_helper.from_array(Wperm.reshape(K, cap, D).astype(np.float32), n("WpermF32")),
        numpy_helper.from_array(Wspec.astype(np.float32), n("WspecF32")),
        _i64(n("Vmap"), Vmap),
        _i64(n("spec_ids"), special_ids),
        _i64(n("Pk"), [P]),
        _i64(n("ax0"), [0]),
        _i64(n("shp_pcD"), [P * cap, D]),
        _i64(n("shp_pc"), [P * cap]),
        _i64(n("shp_1N"), [1, N]),
        numpy_helper.from_array(np.full((1, V), NEG, np.float32), n("base")),
    ]
    return nodes, inits


def quantize_stage1(model, stage1, block_size=128, acc_level=4, pfx="fh_"):
    """Rewrite only the stage-1 MatMul (`<pfx>sims_mm`) to MatMulNBits (W4A16 for int4, W8A16
    for int8). The dense body is already MatMulNBits; stage-2 stays fp32. `block_size` is the
    quant group size along the contraction dim (must divide D; e.g. 32/64/128). Returns ModelProto."""
    from onnxruntime.quantization import matmul_nbits_quantizer as mnq
    bits = {"int4": 4, "int8": 8}[stage1]
    q = mnq.MatMulNBitsQuantizer(model, bits=bits, block_size=block_size, is_symmetric=True,
                                 accuracy_level=acc_level, nodes_to_include=[pfx + "sims_mm"])
    q.process()
    return q.model.model


def build_standalone(npz_path, P, special_ids, out="flash_standalone.onnx", stage1="int4",
                     block_size=128):
    z = np.load(npz_path)
    Cnorm, Wperm, Vmap = z["Cnorm"], z["Wperm"], z["Vmap"]
    K, cap, D = Wperm.shape
    V = K * cap
    W = np.load("artifacts/head_W.npy")  # for special rows
    Wspec = W[special_ids].astype(np.float16)
    nodes, inits = make_flash_nodes(Cnorm, Wperm, Vmap, Wspec, special_ids, P, "h", "logits",
                                    stage1=stage1)
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
    special_ids = np.array([151643, 151645], np.int64)  # bos, eos (§6.5)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
    h_in = {}
    hook = model.lm_head.register_forward_hook(lambda m, i, o: h_in.update(h=i[0]))
    ids = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids
    with torch.no_grad():
        dense = model(ids).logits[0].argmax(-1).numpy()
    hook.remove()
    H = h_in["h"][0].float().numpy()

    # A/B stage-1 precision: does quantized centroid scoring hold accuracy vs fp16?
    for stage1 in ("fp16", "int8", "int4"):
        path = build_standalone("artifacts/clusters.npz", P, special_ids, stage1=stage1)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        agree = sum(int(sess.run(["logits"], {"h": H[i : i + 1].astype(np.float32)})[0].argmax() == dense[i])
                    for i in range(H.shape[0]))
        logger.info(f"stage1={stage1:4s}  flash vs dense argmax: {agree}/{H.shape[0]} = {agree / H.shape[0]:.1%}")


if __name__ == "__main__":
    main()
