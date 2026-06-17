"""Head-precision quality matrix — is FlashHead worth it vs just int4-ing the dense head?

For each candidate head, on the SAME real WikiText-2 hidden states (HF fp32, the input to lm_head):
  top-1 agreement — fraction of positions whose argmax matches the true fp32 head, and
  PPL            — teacher-forced WikiText-2 perplexity.
Reference is the full-precision head_W.npy (fp32). Dense quant heads run their **actual** ORT
kernel (the head subgraph extracted from each spliced model — no numpy re-quant), so the number
is the shipped artifact's quality. FlashHead reuses the pure-numpy reference in eval/{agreement,ppl}.

Usage: `uv run turbohead-head-quality [--max-tokens 2000] [-P 256]`."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger

from turbohead.eval.agreement import wikitext
from turbohead.eval.ppl import collect, flash_ppl
from turbohead.surgery.splice import find_head, DEFAULT_SRC

VARIANTS = ("head16", "head8g128", "head4g128", "head4g32")  # dense-head precisions to compare


def head_session(model_dir):
    """Extract the head subgraph (hidden -> logits) from a spliced model and load it as its own
    session — runs the artifact's real head kernel (MatMul fp16 / MatMulNBits) in isolation."""
    src = f"{model_dir}/model.onnx"
    _, hidden3d = find_head(onnx.load(src, load_external_data=False).graph)
    out = tempfile.mkdtemp()
    sub = f"{out}/head.onnx"
    # check_model=False: the checker rejects genai's com.microsoft contrib ops (docs/ORT_QUIRKS.md).
    onnx.utils.extract_model(src, sub, [hidden3d], ["logits"], check_model=False)
    sess = ort.InferenceSession(sub, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def dense_eval(sess, in_name, H, targets, ref_argmax, chunk=200):
    """argmax agreement vs ref + teacher-forced PPL, running the real head kernel in chunks
    (full (T,V) logits would be ~1GB at T=2000)."""
    agree = 0
    nll = np.empty(len(targets))
    for s in range(0, len(H), chunk):
        h = H[s : s + chunk].astype(np.float32)[None]            # (1, c, D)
        z = sess.run(["logits"], {in_name: h})[0][0]             # (c, V)
        am = z.argmax(-1)
        for j in range(len(h[0])):
            i = s + j
            agree += int(am[j] == ref_argmax[i])
            if i < len(targets):
                row = z[j] - z[j].max()
                nll[i] = np.log(np.exp(row).sum()) - row[targets[i]]
    return agree / len(H), float(np.exp(nll.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC, help="model root dir; variants are <src>/<variant>")
    ap.add_argument("--npz", default="artifacts/clusters.npz")
    ap.add_argument("--head", default="artifacts/head_W.npy")
    ap.add_argument("--model", default=None,
                    help="HF id for the source hidden states (default: read <src>/hf_model_id.txt)")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("-P", "--probes", type=int, default=256)
    a = ap.parse_args()

    # The HF model supplies the real lm_head-input hidden states; it MUST match this artifact's
    # dims. build_all records the id in the artifact dir so the eval is self-describing.
    model_id = a.model or Path(f"{a.src}/hf_model_id.txt").read_text().strip()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    H, targets = collect(model, tok, wikitext(), a.max_tokens)   # H:(T,D)  targets:(T,) next ids
    W = np.load(a.head).astype(np.float32)                       # (V,D) true fp32 head
    ref_argmax = (H @ W.T).argmax(-1)                            # fp32 greedy reference
    logger.info(f"{len(H)} WikiText-2 positions | {model_id} | reference = fp32 head_W")

    logger.info(f"  {'head':14s}{'top-1 agree':>14}{'PPL':>10}")
    for v in VARIANTS:
        d = f"{a.src}/{v}"
        if not Path(f"{d}/model.onnx").exists():
            logger.warning(f"  {v:14s}  (missing — build with turbohead-quantize-head)")
            continue
        sess, in_name = head_session(d)
        agree, ppl = dense_eval(sess, in_name, H, targets, ref_argmax)
        logger.info(f"  {v:14s}{agree:13.1%}{ppl:10.3f}")

    # FlashHead (pure-numpy reference, probed-softmax PPL) — the deploy distribution.
    z = np.load(a.npz)
    Cnorm, Wperm, Vmap = z["Cnorm"], z["Wperm"], z["Vmap"]
    from turbohead.eval.agreement import flash_top1
    f_agree = (flash_top1(H, Cnorm, Wperm, Vmap, a.probes) == ref_argmax).mean()
    f_full, f_cov, cov = flash_ppl(H, Cnorm, Wperm, Vmap, targets, a.probes)
    logger.info(f"  {'flash P=' + str(a.probes):14s}{f_agree:13.1%}{f_full:10.3f}"
                f"   (covered PPL {f_cov:.3f}, coverage {cov:.1%})")


if __name__ == "__main__":
    main()
