"""Dump the bf16 LM-head weight (tied = embed_tokens.weight) to artifacts/head_W.npy for clustering.
Prefer the HF checkpoint over the quantized graph: no dequant error. Usage: `uv run turbohead-extract-head`."""

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from loguru import logger

MODEL = "Qwen/Qwen3-0.6B"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/head_W.npy")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    W = m.lm_head.weight  # the actual head; == embed_tokens.weight when tied
    tied = m.lm_head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    np.save(a.out, W.detach().float().numpy())  # (V, D) fp32
    logger.info(f"saved {a.out}  shape {tuple(W.shape)}  ({'tied' if tied else 'untied'} embeddings)")


if __name__ == "__main__":
    main()
