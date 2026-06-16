"""Phase 3 — splice FlashHead into the genai model (§6, contract A).
Slice last-position hidden -> flash subgraph -> (1,1,V) logits, drop dense head node
(keep its weight; tied embed still needs it). External-data save.
Usage: `uv run turbohead-splice [-P 256] [--stage1 int4] [--block-size 128]`."""

import argparse
import shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper
from loguru import logger
from turbohead.surgery.build_subgraph import make_flash_nodes, quantize_stage1

SRC = "artifacts/qwen3_0_6b_int4_cpu"
DST = "artifacts/qwen3_0_6b_flash"
HEAD = "/lm_head/MatMul_Q8"


def splice(P, special_ids, stage1="int4", block_size=128):
    z = np.load("artifacts/clusters.npz")
    Cnorm, Wperm, Vmap = z["Cnorm"], z["Wperm"], z["Vmap"]
    Wspec = np.load("artifacts/head_W.npy")[special_ids].astype(np.float16)

    Path(DST).mkdir(exist_ok=True)
    for f in ("genai_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")

    m = onnx.load(f"{SRC}/model.onnx")
    g = m.graph
    head = next(n for n in g.node if n.name == HEAD)
    hidden3d = head.input[0]  # (1, seq, D)
    g.node.remove(head)  # weight initializer stays (tied embed uses it)

    # last-position slice: Gather(hidden, -1, axis=1) -> (1, D)
    slice_nodes = [
        helper.make_node("Gather", [hidden3d, "fh_lastidx"], ["fh_hlast"], axis=1),  # (1,D)
    ]
    flash_nodes, flash_inits = make_flash_nodes(
        Cnorm, Wperm, Vmap, Wspec, special_ids, P, "fh_hlast", "fh_logits2d", stage1=stage1
    )
    reshape = helper.make_node("Reshape", ["fh_logits2d", "fh_shp_11V"], ["logits"])

    g.node.extend(slice_nodes + flash_nodes + [reshape])
    g.initializer.extend(flash_inits)
    g.initializer.append(numpy_helper.from_array(np.array(-1, np.int64), "fh_lastidx"))
    g.initializer.append(numpy_helper.from_array(np.array([1, 1, Wperm.shape[0] * Wperm.shape[1]], np.int64), "fh_shp_11V"))

    if stage1 in ("int4", "int8"):
        m = quantize_stage1(m, stage1, block_size)  # fh_sims_mm -> MatMulNBits (body already NBits)

    # ponytail: skip onnx.checker — it rejects genai's com.microsoft contrib ops; ORT load validates.
    for f in ("model.onnx", "model.onnx.data"):
        Path(f"{DST}/{f}").unlink(missing_ok=True)
    onnx.save(m, f"{DST}/model.onnx", save_as_external_data=True, location="model.onnx.data")
    logger.info(f"spliced -> {DST}/model.onnx  (P={P}, stage1={stage1}, block_size={block_size})")


def main():
    ap = argparse.ArgumentParser(description="Splice TurboHead into the baseline ONNX model")
    ap.add_argument("-P", "--probes", type=int, default=256)
    ap.add_argument("--stage1", default="int4", choices=["fp16", "int8", "int4"],
                    help="stage-1 centroid-scoring precision (int4 default, fastest)")
    ap.add_argument("--block-size", type=int, default=128, help="MatMulNBits quant group size")
    a = ap.parse_args()
    splice(a.probes, np.array([151643, 151645], np.int64), stage1=a.stage1, block_size=a.block_size)


if __name__ == "__main__":
    main()
