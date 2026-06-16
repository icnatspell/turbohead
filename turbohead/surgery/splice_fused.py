"""Prototype: contract-B fused splice (greedy only).

Stage-1 (int4 centroid scoring + TopK) stays in the graph; the whole of stage-2
(gather candidate rows -> dot with h -> argmax incl. specials) collapses into one
`turbohead.FlashHeadGreedy` custom op. The graph emits the next-token id directly
(contract B): no (P*cap,D) materialization, no (1,V) logits, no ScatterElements.

Needs csrc/libturbohead.so (build: see csrc/turbohead_op.cc). The .so is copied into
the model dir; the decode loop auto-registers it. Greedy only — no sampling logits.
Usage: `uv run python -m turbohead.surgery.splice_fused [-P 256]`.
"""

import argparse
import shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from loguru import logger
from turbohead.surgery.build_subgraph import quantize_stage1

SRC = "artifacts/qwen3_0_6b_int4_cpu"
DST = "artifacts/qwen3_0_6b_fused"
HEAD = "/lm_head/MatMul_Q8"
OP_LIB = "csrc/libturbohead.so"


def splice_fused(P, special_ids, block_size=128):
    z = np.load("artifacts/clusters.npz")
    Cnorm, Wperm, Vmap = z["Cnorm"], z["Wperm"], z["Vmap"]
    K, cap, D = Wperm.shape
    Wspec = np.load("artifacts/head_W.npy")[special_ids].astype(np.float32)

    Path(DST).mkdir(exist_ok=True)
    for f in ("genai_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
    shutil.copy(OP_LIB, f"{DST}/libturbohead.so")  # decode loop auto-registers this

    m = onnx.load(f"{SRC}/model.onnx")
    g = m.graph
    head = next(n for n in g.node if n.name == HEAD)
    hidden3d = head.input[0]  # (1, seq, D)
    g.node.remove(head)  # weight initializer stays (tied embed uses it)

    nodes = [
        helper.make_node("Gather", [hidden3d, "fh_lastidx"], ["fh_hlast"], axis=1),  # (1,D)
        helper.make_node("MatMul", ["fh_hlast", "fh_Cnorm"], ["fh_sims"], name="fh_sims_mm"),  # ->MatMulNBits
        helper.make_node("TopK", ["fh_sims", "fh_Pk"], ["fh_tv", "fh_ti"], axis=1, sorted=0),
        helper.make_node("Squeeze", ["fh_ti", "fh_ax0"], ["fh_ti1"]),  # (1,P)->(P,)
        helper.make_node("FlashHeadGreedy",
                         ["fh_hlast", "fh_ti1", "fh_Wperm", "fh_Vmap", "fh_Wspec", "fh_spec"],
                         ["next_token"], domain="turbohead"),
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

    # contract B: drop the `logits` output, keep present.* KV, emit next_token (1,1) int64
    kept = [o for o in g.output if o.name != "logits"]
    del g.output[:]
    g.output.extend(kept)
    g.output.append(helper.make_tensor_value_info("next_token", TensorProto.INT64, [1, 1]))
    m.opset_import.append(helper.make_opsetid("turbohead", 1))

    m = quantize_stage1(m, "int4", block_size)  # fh_sims_mm -> MatMulNBits

    for f in ("model.onnx", "model.onnx.data"):
        Path(f"{DST}/{f}").unlink(missing_ok=True)
    onnx.save(m, f"{DST}/model.onnx", save_as_external_data=True, location="model.onnx.data")
    logger.info(f"fused -> {DST}/model.onnx  (P={P}, contract B, greedy)")


def main():
    ap = argparse.ArgumentParser(description="Contract-B fused splice (greedy prototype)")
    ap.add_argument("-P", "--probes", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=128)
    a = ap.parse_args()
    splice_fused(a.probes, np.array([151643, 151645], np.int64), block_size=a.block_size)


if __name__ == "__main__":
    main()
