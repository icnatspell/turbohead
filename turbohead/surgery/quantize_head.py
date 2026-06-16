"""Build a dense-LM-head baseline at a chosen precision — the comparison point for TurboHead.

Replaces the head node of a quantized genai model with a fresh head rebuilt from the
full-precision head weight, at {16, 8, 4}-bit and a chosen MatMulNBits group size. Body and
everything else untouched, so this isolates the effect of head precision. Generalizes to any
genai export (auto-detects the head via find_head).

  16-bit = exact reference (no quantization error). On CPU it folds to fp32 at load — there's no
           fp16 matmul kernel — so 16-bit and 32-bit behave identically at runtime (see
           docs/ORT_QUIRKS.md); we store fp16 for honest disk size.
  8/4-bit = MatMulNBits (W8A16/W4A16), the same fast fused-dequant gemv the model body uses.

Usage: `uv run turbohead-quantize-head --bits 4 --group-size 128 [--src DIR] [--dst DIR]`.
"""

import argparse
import shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from loguru import logger
from turbohead.surgery.splice import find_head, DEFAULT_SRC, HEAD_W

CFG = ("genai_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _nbits(model, bits, block_size, node_name):
    from onnxruntime.quantization import matmul_nbits_quantizer as mnq
    q = mnq.MatMulNBitsQuantizer(model, bits=bits, block_size=block_size, is_symmetric=True,
                                 accuracy_level=4, nodes_to_include=[node_name])
    q.process()
    return q.model.model


def quantize_head(src=DEFAULT_SRC, dst=None, bits=4, group_size=128, head=HEAD_W):
    if bits not in (16, 8, 4):
        raise ValueError(f"bits must be 16, 8 or 4, got {bits}")
    dst = dst or f"{src}_head{bits}" + (f"g{group_size}" if bits < 16 else "")

    Path(dst).mkdir(parents=True, exist_ok=True)
    for f in CFG:
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")

    m = onnx.load(f"{src}/model.onnx")
    g = m.graph
    head_node, hidden3d = find_head(g)  # (1, seq, D)
    g.node.remove(head_node)  # old quantized head weight initializer stays (tied embed uses it)

    Wt = np.load(head).T  # (D, V) for hidden @ W -> logits
    if bits == 16:
        g.initializer.append(numpy_helper.from_array(Wt.astype(np.float16), "lm_head_W16"))
        g.node.append(helper.make_node("Cast", ["lm_head_W16"], ["lm_head_Wf"], to=TensorProto.FLOAT))
        g.node.append(helper.make_node("MatMul", [hidden3d, "lm_head_Wf"], ["logits"], name="lm_head"))
    else:
        g.initializer.append(numpy_helper.from_array(Wt.astype(np.float32), "lm_head_W"))
        g.node.append(helper.make_node("MatMul", [hidden3d, "lm_head_W"], ["logits"], name="lm_head"))
        m = _nbits(m, bits, group_size, "lm_head")  # -> MatMulNBits

    for f in ("model.onnx", "model.onnx.data"):
        Path(f"{dst}/{f}").unlink(missing_ok=True)
    onnx.save(m, f"{dst}/model.onnx", save_as_external_data=True, location="model.onnx.data")
    tag = f"{bits}-bit" + (f" g{group_size}" if bits < 16 else " (exact)")
    logger.info(f"head -> {dst}/model.onnx  ({tag})")
    return dst


def main():
    ap = argparse.ArgumentParser(description="Build a dense-LM-head baseline at a chosen precision")
    ap.add_argument("--src", default=DEFAULT_SRC, help="baseline genai ONNX dir")
    ap.add_argument("--dst", default=None, help="output dir (default: <src>_head<bits>[g<group>])")
    ap.add_argument("--head", default=HEAD_W, help="fp32 head_W.npy for this model")
    ap.add_argument("--bits", type=int, default=4, choices=[16, 8, 4])
    ap.add_argument("--group-size", type=int, default=128, help="MatMulNBits group size (8/4-bit)")
    a = ap.parse_args()
    quantize_head(a.src, a.dst, a.bits, a.group_size, a.head)


if __name__ == "__main__":
    main()
