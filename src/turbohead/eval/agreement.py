"""Phase 1/2 exit gate — top-1 agreement of FlashHead vs dense head on REAL hidden states.
Hooks lm_head input on the HF model over real text; flash retrieval uses artifacts/clusters.npz.
Usage: `uv run turbohead-agreement [--npz artifacts/clusters.npz]`."""

import argparse
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


def _detokenize(t):
    """Undo WikiText-2's tokenized markup (`@-@`, spaced punctuation, ` \\n `). Raw WikiText inflates
    teacher-forced PPL wildly and unevenly across models — e.g. LFM2.5-350M scores ~680 raw vs ~116
    detokenized — which made the PPL column noise. Standard detok before scoring restores meaningful,
    more comparable numbers. ponytail: the high-impact substitutions, not a full reverse-tokenizer."""
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([.,;:!?)\]])", r"\1", t).replace(" \n ", "\n").replace("( ", "(")
    return t


def wikitext(n_chars=20000):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = _detokenize("".join(t for t in ds["text"] if t.strip()))
    return text[:n_chars]


def collect_hidden(model, tok, text, max_tokens=2000):
    h_in = {}
    hook = model.lm_head.register_forward_hook(lambda m, i, o: h_in.update(h=i[0]))
    ids = tok(text, return_tensors="pt").input_ids[:, :max_tokens]
    with torch.no_grad():
        logits = model(ids).logits[0]  # (T, V)
    hook.remove()
    H = h_in["h"][0].float().numpy()  # (T, D) input to lm_head
    dense = logits.argmax(-1).numpy()  # (T,) true greedy token
    return H, dense


def flash_top1(H, Cnorm, Wperm, Vmap, P):
    K, cap, D = Wperm.shape
    Cn, Wp = Cnorm.astype(np.float32), Wperm.astype(np.float32)
    sims = H @ Cn  # (T, K)
    out = np.empty(H.shape[0], np.int64)
    for i, h in enumerate(H):
        top = np.argpartition(sims[i], -P)[-P:]
        slot = (Wp[top].reshape(P * cap, D) @ h).argmax()
        out[i] = Vmap[top].reshape(-1)[slot]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="artifacts/clusters.npz")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    a = ap.parse_args()
    npz = np.load(a.npz)
    Cnorm, Wperm, Vmap = npz["Cnorm"], npz["Wperm"], npz["Vmap"]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext())
    logger.info(f"{H.shape[0]} real WikiText-2 positions")
    for P in (128, 256, 384, 512):
        agree = (flash_top1(H, Cnorm, Wperm, Vmap, P) == dense).mean()
        logger.info(f"P={P:<4d} top-1 agreement vs dense: {agree:.1%}")


if __name__ == "__main__":
    main()
