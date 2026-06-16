"""Fused splice (contract H — candidate shortlist out).

Stage-1 (int4 centroid scoring + TopK) stays in the graph; the whole of stage-2
(gather candidate rows -> dot with h) collapses into one `turbohead.FlashHeadSelect`
custom op. The graph emits the candidate shortlist `cand_logits (1,N)` + `cand_ids
(1,N)` — no (P*cap,D) materialization, no (1,V) logits, no ScatterElements. The decode
loop does greedy (argmax) or sampling (softmax) over the shortlist, so temperature>0
works without ever forming the full vocab vector.

Needs csrc/libturbohead.so (build: bash csrc/build.sh). The .so is copied into the
model dir; the decode loop auto-registers it.
Usage: `uv run python -m turbohead.surgery.splice_fused [-P 256]`.
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from loguru import logger
from turbohead.surgery.build_subgraph import quantize_stage1

OP_LIB = "csrc/libturbohead.so"
MATMUL_OPS = {"MatMul", "MatMulNBits", "Gemm", "FusedMatMul"}


def find_head(g):
    """The dense head is whatever node produces the `logits` graph output (MatMulNBits for
    an int4/int8 head, MatMul for fp16 — works across genai exports). Returns (node, hidden)."""
    prod = {o: n for n in g.node for o in n.output}
    node = prod.get("logits")
    if node is None or node.op_type not in MATMUL_OPS:
        raise RuntimeError(f"no matmul head feeding 'logits' (found {node and node.op_type}); "
                           "this expects a genai-style export where the head emits logits directly")
    return node, node.input[0]  # input[0] = (1, seq, D) hidden state


def read_eos(src):
    """EOS (+BOS) ids from genai_config -> always-scored special rows, so greedy can emit them."""
    eos = json.load(open(f"{src}/genai_config.json"))["model"].get("eos_token_id", [])
    return sorted({*(eos if isinstance(eos, list) else [eos])})


def splice_fused(src, dst, P, block_size=128):
    z = np.load("artifacts/clusters.npz")
    Cnorm, Wperm, Vmap = z["Cnorm"], z["Wperm"], z["Vmap"]
    K, cap, D = Wperm.shape
    special_ids = np.asarray(read_eos(src), np.int64)
    S = len(special_ids)
    N = P * cap + S
    Wspec = np.load("artifacts/head_W.npy")[special_ids].astype(np.float32)

    Path(dst).mkdir(parents=True, exist_ok=True)
    for f in ("genai_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    shutil.copy(OP_LIB, f"{dst}/libturbohead.so")  # decode loop auto-registers this

    m = onnx.load(f"{src}/model.onnx")
    g = m.graph
    head, hidden3d = find_head(g)  # (1, seq, D)
    g.node.remove(head)  # weight initializer stays (tied embed uses it)

    nodes = [
        helper.make_node("Gather", [hidden3d, "fh_lastidx"], ["fh_hlast"], axis=1),  # (1,D)
        helper.make_node("MatMul", ["fh_hlast", "fh_Cnorm"], ["fh_sims"], name="fh_sims_mm"),  # ->MatMulNBits
        helper.make_node("TopK", ["fh_sims", "fh_Pk"], ["fh_tv", "fh_ti"], axis=1, sorted=0),
        helper.make_node("Squeeze", ["fh_ti", "fh_ax0"], ["fh_ti1"]),  # (1,P)->(P,)
        helper.make_node("FlashHeadSelect",
                         ["fh_hlast", "fh_ti1", "fh_Wperm", "fh_Vmap", "fh_Wspec", "fh_spec"],
                         ["cand_logits", "cand_ids"], domain="turbohead"),
    ]
    inits = [
        numpy_helper.from_array(Cnorm.astype(np.float32), "fh_Cnorm"),
        numpy_helper.from_array(Wperm.reshape(K, cap, D).astype(np.float32), "fh_Wperm"),
        numpy_helper.from_array(Vmap.astype(np.int64), "fh_Vmap"),
        numpy_helper.from_array(Wspec, "fh_Wspec"),
        numpy_helper.from_array(np.asarray(special_ids, np.int64), "fh_spec"),
        numpy_helper.from_array(np.array([P], np.int64), "fh_Pk"),
        numpy_helper.from_array(np.array([0], np.int64), "fh_ax0"),
        numpy_helper.from_array(np.array(-1, np.int64), "fh_lastidx"),
    ]
    g.node.extend(nodes)
    g.initializer.extend(inits)

    # contract H: drop the `logits` output, keep present.* KV, emit the shortlist
    kept = [o for o in g.output if o.name != "logits"]
    del g.output[:]
    g.output.extend(kept)
    g.output.append(helper.make_tensor_value_info("cand_logits", TensorProto.FLOAT, [1, N]))
    g.output.append(helper.make_tensor_value_info("cand_ids", TensorProto.INT64, [1, N]))
    m.opset_import.append(helper.make_opsetid("turbohead", 1))

    m = quantize_stage1(m, "int4", block_size)  # fh_sims_mm -> MatMulNBits

    for f in ("model.onnx", "model.onnx.data"):
        Path(f"{dst}/{f}").unlink(missing_ok=True)
    onnx.save(m, f"{dst}/model.onnx", save_as_external_data=True, location="model.onnx.data")
    logger.info(f"fused -> {dst}/model.onnx  (P={P}, contract H, N={N}, eos={special_ids.tolist()})")


def main():
    ap = argparse.ArgumentParser(description="Fused splice (contract H — shortlist out)")
    ap.add_argument("--src", default="artifacts/qwen3_0_6b_int4_cpu", help="baseline genai ONNX dir")
    ap.add_argument("--dst", default="artifacts/qwen3_0_6b_fused", help="output spliced dir")
    ap.add_argument("-P", "--probes", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=128)
    a = ap.parse_args()
    splice_fused(a.src, a.dst, a.probes, block_size=a.block_size)


if __name__ == "__main__":
    main()
