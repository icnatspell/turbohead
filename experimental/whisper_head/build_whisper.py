"""Build the TurboHead splice for one Whisper model, end to end -> an export dir with dense +
flash (onnx, fused) decoders ready for whisper_decode.py.

Whisper is encoder-decoder, so the head lives in `decoder.onnx`. The surgery core is reused unchanged
(`build_clusters`, `splice.splice`); the only Whisper-specific bit is pulling `proj_out` for the head.
Steps per model:
  1. genai builder  -> export/{encoder,decoder}.onnx (+ .data, configs), int4 body
  2. extract proj_out (tied embed) -> export/head_W.npy
  3. balanced k-means -> export/clusters.npz   (cap must divide V=51865; default 41)
  4. stage decoder.onnx as model.onnx (splice loads `model.onnx`; external data ref stays valid)
  5. splice --backend {onnx,fused} -> export_onnx/ , export_fused/  (each self-contained, encoder copied in)

    uv run python experimental/whisper_head/build_whisper.py openai/whisper-tiny whisper_tiny [--cap 41] [-P 256]

Idempotent-ish: rebuilds in place. Heavy (torch + genai); needs `uv sync --extra surgery`.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from loguru import logger
from onnx import helper, numpy_helper
from transformers import WhisperForConditionalGeneration

from turbohead.surgery.build_clusters import build, kmeans
from turbohead.surgery.quantize_head import quantize_head
from turbohead.surgery.splice import splice

ROOT = Path("artifacts")


def export_genai(model_id, out):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "src/turbohead/surgery/_genai_build.py",
                    "-m", model_id, "-o", str(out), "-p", "int4", "-e", "cpu",
                    "-c", "artifacts/hf_cache", "--extra_options",
                    "int4_block_size=128", "int4_accuracy_level=4", "hf_token=false"], check=True)
    (out / "source_model.txt").write_text(model_id + "\n")  # whisper_decode reads this for the processor


def make_fp32_head(exp, W):
    """Dense fp32-head baseline decoder: swap the int4 MatMulNBits logits node for a plain fp32 MatMul
    on the real head weight. This is the "fp32-equivalent dense head" RESULTS.md headlines flash against
    (genai int4-quantizes the head by default — the hardest, cheapest baseline; report both)."""
    m = onnx.load(str(exp / "decoder.onnx"))  # pulls decoder.onnx.data
    g = m.graph
    head = {o: n for n in g.node for o in n.output}["logits"]
    hidden = head.input[0]  # io_dtype is fp32 on the CPU int4 build, so fp32 weight matches (no cast)
    g.node.remove(head)
    g.initializer.append(numpy_helper.from_array(W.T.astype(np.float32), "fh_dense_W"))  # (D,V)
    g.node.append(helper.make_node("MatMul", [hidden, "fh_dense_W"], ["logits"]))
    onnx.save(m, str(exp / "decoder_f32.onnx"), save_as_external_data=True,
              location="decoder_f32.onnx.data")
    logger.info("dense fp32-head baseline -> decoder_f32.onnx")


def extract_head(model_id, out):
    m = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    W = m.get_output_embeddings().weight  # proj_out == tied embed, (V, D)
    np.save(out / "head_W.npy", W.detach().float().numpy())
    logger.info(f"head_W {tuple(W.shape)} saved")
    return W.shape[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_id")
    ap.add_argument("slug")
    ap.add_argument("--cap", type=int, default=41, help="tokens/cluster; must divide V=51865 (41|55|23)")
    ap.add_argument("-P", "--probes", type=int, default=256)
    a = ap.parse_args()

    exp = ROOT / a.slug / "export"
    if not (exp / "decoder.onnx").exists():
        logger.info(f"exporting {a.model_id} -> {exp}")
        export_genai(a.model_id, exp)
    else:
        logger.info(f"reusing export {exp}")
        (exp / "source_model.txt").write_text(a.model_id + "\n")

    V = extract_head(a.model_id, exp)
    assert V % a.cap == 0, f"cap {a.cap} must divide V={V}"
    K = V // a.cap
    logger.info(f"clustering V={V} -> K={K} x cap={a.cap}")
    W = np.load(exp / "head_W.npy").astype(np.float32)
    _C, homes = kmeans(W, K, a.cap)
    Cnorm, Wperm, Vmap = build(W, homes, a.cap)
    np.savez(exp / "clusters.npz", Cnorm=Cnorm, Wperm=Wperm, Vmap=Vmap)

    make_fp32_head(exp, W)

    # splice loads `<src>/model.onnx`; stage the decoder there (its external-data ref to
    # decoder.onnx.data stays valid in the same dir). The dense decoder.onnx stays for the baseline.
    shutil.copy(exp / "decoder.onnx", exp / "model.onnx")

    # dense-head precision baselines (RESULTS.md-style): fp16=head16, int8=head8g128, int4=head4g128.
    # All rebuilt from the same fp32 head_W via core quantize_head -> only head precision varies.
    head = str(exp / "head_W.npy")
    for bits, g in ((16, 128), (8, 128), (4, 128)):
        name = f"head{bits}" + (f"g{g}" if bits < 16 else "")
        quantize_head(str(exp), str(ROOT / a.slug / name), bits, g, head)

    for backend in ("onnx", "fused"):
        dst = ROOT / a.slug / f"export_{backend}"
        # int8 stage-2 weights (FlashHeadSelectQ8) for fused: ~4x less head read than fp32, quality-
        # neutral, and the only way flash-fused beats an int4 dense head (else stage-2 fp32 reads more
        # bytes than the whole int4 head). onnx backend keeps fp32 (no custom-op int8 path).
        wdt = "int8" if backend == "fused" else "fp32"
        splice(str(exp), str(dst), backend, a.probes, "int4", 128,
               str(exp / "clusters.npz"), head, head_weight_dtype=wdt)
        logger.info(f"spliced {backend} (stage2={wdt}) -> {dst}")

    logger.info(f"done {a.slug}. bench:\n"
                f"  uv run python experimental/whisper_head/whisper_decode.py {exp} --bench  # dense\n"
                f"  uv run python experimental/whisper_head/whisper_decode.py {ROOT/a.slug}/export_onnx "
                f"--decoder model.onnx --bench  # onnx flash\n"
                f"  uv run python experimental/whisper_head/whisper_decode.py {ROOT/a.slug}/export_fused "
                f"--decoder model.onnx --bench  # fused flash")


if __name__ == "__main__":
    main()
