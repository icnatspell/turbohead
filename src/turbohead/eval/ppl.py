"""Quality gate — WikiText-2 perplexity of the dense head vs TurboHead's probed-softmax.

Dense PPL is the standard full-vocab softmax. Flash PPL softmaxes over only the P*cap probed
candidates (the deploy distribution); a target token outside that set has ~0 probability, so we
also report *coverage* = fraction of targets the candidate set contains. PPL uses a tiny floor
for uncovered targets, so a low coverage shows up as inflated flash PPL — read the two together.

Usage: `uv run turbohead-ppl [--npz artifacts/clusters.npz] [-P 256]`.
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from turbohead.eval.agreement import wikitext

EPS = 1e-12  # floor prob for uncovered targets -> finite PPL


def collect(model, tok, text, max_tokens=2000):
    """Hidden states + the actual next-token targets (teacher forcing)."""
    h_in = {}
    hook = model.lm_head.register_forward_hook(lambda m, i, o: h_in.update(h=i[0]))
    ids = tok(text, return_tensors="pt").input_ids[:, :max_tokens]
    with torch.no_grad():
        model(ids)
    hook.remove()
    H = h_in["h"][0].float().numpy()            # (T, D)
    targets = ids[0, 1:].numpy()                # next token for positions 0..T-2
    return H[:-1], targets                       # align: predict targets[i] from H[i]


def dense_ppl(H, W, targets):
    nll = np.empty(len(targets))
    for i, (h, t) in enumerate(zip(H, targets)):
        z = W @ h                                # (V,) full logits
        z -= z.max()
        nll[i] = np.log(np.exp(z).sum()) - z[t]
    return float(np.exp(nll.mean()))


def flash_ppl(H, Cnorm, Wperm, Vmap, targets, P):
    K, cap, D = Wperm.shape
    Cn, Wp = Cnorm.astype(np.float32), Wperm.astype(np.float32)
    sims = H @ Cn                                # (T, K)
    nll = np.empty(len(targets))
    hit_mask = np.zeros(len(targets), bool)
    for i, (h, t) in enumerate(zip(H, targets)):
        top = np.argpartition(sims[i], -P)[-P:]
        ids = Vmap[top].reshape(-1)              # (P*cap,) candidate token ids
        z = Wp[top].reshape(P * cap, D) @ h      # (P*cap,) candidate logits
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        hit = ids == t
        ptrue = p[hit].sum() if hit.any() else 0.0
        hit_mask[i] = hit.any()
        nll[i] = -np.log(max(ptrue, EPS))
    full = float(np.exp(nll.mean()))                    # all targets (uncovered -> floored)
    covered = float(np.exp(nll[hit_mask].mean()))       # ranking quality given coverage
    return full, covered, hit_mask.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="artifacts/clusters.npz")
    ap.add_argument("--head", default="artifacts/head_W.npy")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("-P", "--probes", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=2000)
    a = ap.parse_args()

    z = np.load(a.npz)
    Cnorm, Wperm, Vmap = z["Cnorm"], z["Wperm"], z["Vmap"]
    W = np.load(a.head).astype(np.float32)       # (V, D) dense head
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    H, targets = collect(model, tok, wikitext(), a.max_tokens)
    logger.info(f"{len(targets)} WikiText-2 positions, P={a.probes}")

    d = dense_ppl(H, W, targets)
    f_full, f_cov, cov = flash_ppl(H, Cnorm, Wperm, Vmap, targets, a.probes)
    logger.info(f"dense PPL                {d:.3f}")
    logger.info(f"flash PPL (covered)      {f_cov:.3f}  ranking quality where target is in the probed set")
    logger.info(f"flash PPL (all)          {f_full:.3f}  uncovered targets floored -> coverage-limited")
    logger.info(f"candidate coverage       {cov:.1%}  raise P to improve")


if __name__ == "__main__":
    main()
