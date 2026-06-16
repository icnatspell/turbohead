"""Splice TurboHead's approximate head into a quantized genai ONNX model.

Two interchangeable backends — choose at splice time; the decode loop auto-detects which one a
model uses, so the run command is identical either way:

  onnx  — stage 2 as plain ONNX ops; emits full (1,V) logits (contract A). No native library,
          maximal portability. Use when you need full-vocab logits or can't ship a .so.
  fused — stage 2 as a single custom op; emits the candidate shortlist (contract H). Needs
          csrc/libturbohead.so (build: bash csrc/build.sh). Fastest; greedy + sampling. Default.

Both backends share stage 1 (int4 centroid scoring + TopK) and the same clustering assets; they
differ only in how stage 2 is realized and what the graph outputs. Generalizes to any genai-style
export (auto-detects the head node; reads EOS from genai_config).

Usage: `uv run turbohead-splice [--backend fused|onnx] [--src DIR] [--dst DIR] [-P 256]`.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from loguru import logger
from turbohead.surgery.build_subgraph import (
    stage1_nodes, onnx_stage2_nodes, fused_stage2_nodes, quantize_stage1,
)

DEFAULT_SRC = "artifacts/qwen3_0_6b_int4_cpu"
CLUSTERS = "artifacts/clusters.npz"
HEAD_W = "artifacts/head_W.npy"
OP_LIB = "csrc/libturbohead.so"
MATMUL_OPS = {"MatMul", "MatMulNBits", "Gemm", "FusedMatMul"}
DST_SUFFIX = {"fused": "_fused", "onnx": "_onnx"}


def find_head(graph):
    """The dense head is whatever node produces the `logits` graph output (MatMulNBits for an
    int4/int8 head, MatMul for fp16). Returns (node, hidden_tensor_name). Generalizes across
    genai exports — no node-name matching."""
    prod = {o: n for n in graph.node for o in n.output}
    node = prod.get("logits")
    if node is None or node.op_type not in MATMUL_OPS:
        raise RuntimeError(f"no matmul head feeding 'logits' (found {node and node.op_type}); "
                           "expects a genai-style export where the head emits logits directly")
    return node, node.input[0]  # input[0] = (1, seq, D) hidden state


def copy_configs(src, dst):
    """Copy every non-model file (tokenizer, genai_config, chat template if present, ...) from a
    genai export to a derived dir. Robust across models — base exports omit chat_template.jinja."""
    Path(dst).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(src):
        if f not in ("model.onnx", "model.onnx.data") and os.path.isfile(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", f"{dst}/{f}")


def read_eos(src):
    """EOS (+BOS) ids from genai_config -> always-scored special rows so greedy can emit them."""
    eos = json.load(open(f"{src}/genai_config.json"))["model"].get("eos_token_id", [])
    return sorted({*(eos if isinstance(eos, list) else [eos])})


def splice(src=DEFAULT_SRC, dst=None, backend="fused", P=256, stage1="int4", block_size=128,
           npz=CLUSTERS, head=HEAD_W):
    """Splice the flash head into `src` -> `dst` using `backend` ('fused' or 'onnx').
    `npz`/`head` are the clustering assets for this model (default the Qwen3-0.6B paths)."""
    if backend not in DST_SUFFIX:
        raise ValueError(f"backend must be 'fused' or 'onnx', got {backend!r}")
    dst = dst or f"{src}{DST_SUFFIX[backend]}"
    if backend == "fused" and not Path(OP_LIB).exists():
        raise FileNotFoundError(f"{OP_LIB} not built — run `bash csrc/build.sh` first")

    z = np.load(npz)
    Cnorm, Wperm, Vmap = z["Cnorm"], z["Wperm"], z["Vmap"]
    K, cap, D = Wperm.shape
    V = K * cap
    special_ids = np.asarray(read_eos(src), np.int64)
    N = P * cap + len(special_ids)
    Wspec = np.load(head)[special_ids]  # (S,D) fp32; stage-2 builders cast as needed

    copy_configs(src, dst)
    if backend == "fused":
        shutil.copy(OP_LIB, f"{dst}/libturbohead.so")  # decode loop auto-registers this

    m = onnx.load(f"{src}/model.onnx")
    g = m.graph
    head, hidden3d = find_head(g)  # (1, seq, D)
    g.node.remove(head)  # weight initializer stays (tied embed uses it)

    # shared: slice last-position hidden -> (1,D), then stage 1 (centroid scoring + TopK)
    nodes = [helper.make_node("Gather", [hidden3d, "fh_lastidx"], ["fh_hlast"], axis=1)]
    inits = [numpy_helper.from_array(np.array(-1, np.int64), "fh_lastidx")]
    s1n, s1i, ti1 = stage1_nodes(Cnorm, P, "fh_hlast", stage1=stage1)
    nodes += s1n
    inits += s1i

    if backend == "onnx":
        s2n, s2i = onnx_stage2_nodes(Wperm, Vmap, Wspec, special_ids, P, "fh_hlast", ti1,
                                     "fh_logits2d")
        nodes += s2n + [helper.make_node("Reshape", ["fh_logits2d", "fh_shp_11V"], ["logits"])]
        inits += s2i + [_i64v("fh_shp_11V", [1, 1, V])]
        # `logits` graph output already declared by the export — reused as-is (contract A)
    else:  # fused (contract H): emit the shortlist, drop the (1,V) logits output
        s2n, s2i = fused_stage2_nodes(Wperm, Vmap, Wspec, special_ids, "fh_hlast", ti1)
        nodes += s2n
        inits += s2i
        kept = [o for o in g.output if o.name != "logits"]
        del g.output[:]
        g.output.extend(kept)
        g.output.append(helper.make_tensor_value_info("cand_logits", TensorProto.FLOAT, [1, N]))
        g.output.append(helper.make_tensor_value_info("cand_ids", TensorProto.INT64, [1, N]))
        m.opset_import.append(helper.make_opsetid("turbohead", 1))

    g.node.extend(nodes)
    g.initializer.extend(inits)
    if stage1 in ("int4", "int8"):
        m = quantize_stage1(m, stage1, block_size)  # fh_sims_mm -> MatMulNBits

    # ponytail: skip onnx.checker — it rejects genai's com.microsoft contrib ops; ORT load validates.
    for f in ("model.onnx", "model.onnx.data"):
        Path(f"{dst}/{f}").unlink(missing_ok=True)
    onnx.save(m, f"{dst}/model.onnx", save_as_external_data=True, location="model.onnx.data")
    logger.info(f"spliced [{backend}] -> {dst}/model.onnx  "
                f"(P={P}, stage1={stage1}, block_size={block_size}, eos={special_ids.tolist()})")
    return dst


def _i64v(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.int64), name)


def main():
    ap = argparse.ArgumentParser(description="Splice TurboHead into a quantized genai ONNX model")
    ap.add_argument("--backend", default="fused", choices=["fused", "onnx"],
                    help="fused = custom-op shortlist (contract H, fastest); onnx = portable (1,V) logits (contract A)")
    ap.add_argument("--src", default=DEFAULT_SRC, help="baseline genai ONNX dir")
    ap.add_argument("--dst", default=None, help="output dir (default: <src><_fused|_onnx>)")
    ap.add_argument("--npz", default=CLUSTERS, help="clusters .npz for this model")
    ap.add_argument("--head", default=HEAD_W, help="fp32 head_W.npy for this model")
    ap.add_argument("-P", "--probes", type=int, default=256)
    ap.add_argument("--stage1", default="int4", choices=["fp16", "int8", "int4"],
                    help="stage-1 centroid-scoring precision (int4 default, fastest)")
    ap.add_argument("--block-size", type=int, default=128, help="MatMulNBits quant group size")
    a = ap.parse_args()
    splice(a.src, a.dst, a.backend, a.probes, a.stage1, a.block_size, a.npz, a.head)


if __name__ == "__main__":
    main()
