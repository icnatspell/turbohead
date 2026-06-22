"""Go/no-go gate for FlashHead-style FFN activation sparsity (the MLP analogue of the head splice).

Measures the *oracle* ceiling: per SwiGLU MLP, compute the true intermediate `act(gate(x))*up(x)`,
keep only the top-k magnitude neurons (TEAL-style magnitude sparsity), zero the rest, run down_proj.
Reports end-to-end top-1 agreement vs the full model over real WikiText-2 — so cross-layer drift
compounds naturally (FFN sparsity approximates in *every* layer, unlike the head's one splice).

This is an UPPER BOUND: a real deploy predicts the active set before computing gate/up (SparseInfer
sign-bit predictor) and can't see the true magnitudes. If the oracle ceiling here doesn't clear the
bar, the predictor-limited reality won't either — stop before writing any int4-sparse kernel.

Usage: `uv run python experimental/ffn_sparsity/ffn_sparsity_poc.py [--model Qwen/Qwen3-0.6B]`."""

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from turbohead.eval.agreement import wikitext


def topk_mask(z, keep):
    """Keep the `keep` largest-|.| entries per row of z (last dim), zero the rest. Fixed-k → static
    shape downstream (the implementation tip: top-k beats threshold for ORT). ponytail: torch.topk,
    no custom sort."""
    if keep >= z.shape[-1]:
        return z
    k = max(1, int(round(keep)))
    idx = z.abs().topk(k, dim=-1).indices
    out = torch.zeros_like(z)
    return out.scatter_(-1, idx, z.gather(-1, idx))


def patch_mlp(mlp, keep_frac):
    """Wrap a SwiGLU MLP's forward to apply oracle top-k magnitude sparsity on the intermediate."""
    g, u, d, act = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
    F = g.out_features
    keep = keep_frac * F

    def forward(x):
        inter = act(g(x)) * u(x)
        return d(topk_mask(inter, keep))

    mlp.forward = forward


def agreement_at(model, ids, dense, keep_frac):
    mlps = [m for n, m in model.named_modules() if n.endswith("mlp") and hasattr(m, "gate_proj")]
    saved = [m.forward for m in mlps]
    for m in mlps:
        patch_mlp(m, keep_frac)
    try:
        with torch.no_grad():
            pred = model(ids).logits[0].argmax(-1).numpy()
    finally:
        for m, f in zip(mlps, saved):
            m.forward = f
    return (pred == dense).mean(), len(mlps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-tokens", type=int, default=2000)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    ids = tok(wikitext(), return_tensors="pt").input_ids[:, : a.max_tokens]
    with torch.no_grad():
        dense = model(ids).logits[0].argmax(-1).numpy()
    logger.info(f"{dense.shape[0]} real WikiText-2 positions, model={a.model}")
    for keep in (1.0, 0.7, 0.6, 0.5, 0.4, 0.3):
        agree, n = agreement_at(model, ids, dense, keep)
        logger.info(f"keep={keep:.0%}  sparsity={1-keep:.0%}  top-1 agreement={agree:.1%}  ({n} MLPs)")


if __name__ == "__main__":
    main()
