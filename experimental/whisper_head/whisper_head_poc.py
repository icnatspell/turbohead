"""whisper_head — go/no-go gate for FlashHead on openai/whisper-small.

Whisper is encoder-decoder ASR, not a causal text LM, so the *runtime harness* (ONNX export,
audio front-end, encoder pass, cross-attn KV) is real new work. This POC answers the only two
questions that decide whether that work is worth starting — WITHOUT building any of it, straight
from torch/HF (the same shortcut ffn_sparsity_poc.py takes):

  1. HEAD SHARE — is `proj_out` a big enough slice of a *decode step* to clear the Amdahl ceiling?
     Encoder runs once and amortizes, so per-step cost = decoder layers + head. Reported as the
     memory-bound byte/param share at M=1 (the proxy behind our Qwen3 "head = 24.8% of decode" gate).
  2. RECALL — does the clustering head actually route Whisper's decoder hidden states? Top-1
     agreement vs the dense head on REAL hidden states from transcribing real audio. This is the
     novel risk: Whisper's hidden-state geometry is nothing like a text LM's.

Reuses core unchanged: `surgery.build_clusters.{kmeans,build}` for the partition,
`eval.agreement.flash_top1` for retrieval. Only the audio/encoder-decoder collection is new.

    uv run python experimental/whisper_head/whisper_head_poc.py [--clips 12] [--cap 41]

~2-3 min on CPU (downloads whisper-small ~1GB + librispeech dummy on first run).
keep the dense head (cap=V, P=1 covers all) -> 100% agreement is the implicit self-check.
"""

import argparse

import numpy as np
import torch
from loguru import logger
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from turbohead.surgery.build_clusters import build, kmeans

MODEL = "openai/whisper-small"


def flash_topk(H, Cnorm, Wperm, Vmap, P, k=5):
    """flash retrieval returning each position's top-k candidate token ids (unordered). top-1 == the
    argmax flash would emit. Used for recall@1 (top-1 agreement) and recall@k (is the dense pick in the
    flash top-k) — the latter says how much task headroom the approximation keeps."""
    K, cap, D = Wperm.shape
    Cn, Wp = Cnorm.astype(np.float32), Wperm.astype(np.float32)
    sims = H @ Cn
    out = np.empty((len(H), k), np.int64)
    for i, h in enumerate(H):
        top = np.argpartition(sims[i], -P)[-P:]
        cand_logits = Wp[top].reshape(-1, D) @ h
        cand_ids = Vmap[top].reshape(-1)
        out[i] = cand_ids[np.argpartition(cand_logits, -k)[-k:]]
    return out  # (T, k)


def pick_cap(V, target=41):
    """cap must divide V exactly (no padding — same constraint as core). Return the divisor of V
    closest to `target`; smaller cap => more, tighter clusters (better recall) but a bigger stage-1
    gemv. whisper-small V=51865=5*11*23*41 -> {23,41,55} are the sane options."""
    divs = [d for d in range(2, V) if V % d == 0 and 2 <= V // d <= 8000]
    return min(divs, key=lambda d: abs(d - target))


def collect_whisper_hidden(model, proc, clips, max_new=64):
    """Real decode-time hidden states: transcribe each audio clip with greedy generate and hook the
    input to `proj_out` at every step. Returns H (N,D) and the dense greedy token id per position —
    exactly the distribution a deployed flash head would see. (proj_out input == decoder final hidden
    state; output argmax == the token the dense head picks, our agreement target.)"""
    H_in, dense = [], []

    def hook(_m, inp, out):
        h = inp[0].reshape(-1, inp[0].shape[-1])  # (steps, D); prefill step has the forced prefix
        H_in.append(h.float().numpy())
        dense.append(out.reshape(-1, out.shape[-1]).argmax(-1).numpy())

    handle = model.proj_out.register_forward_hook(hook)
    with torch.no_grad():
        for wav, sr in clips:
            feats = proc(wav, sampling_rate=sr, return_tensors="pt").input_features
            model.generate(feats, max_new_tokens=max_new, do_sample=False, num_beams=1)
    handle.remove()
    return np.concatenate(H_in), np.concatenate(dense)


def head_share(model):
    """Memory-bound (byte) share of `proj_out` within a single decode step. Counts decoder.layers as
    the per-step work; encoder runs once (amortized) and embed_tokens is a lookup, both excluded.
    ponytail: cross-attn k/v/proj are cached after step 0, so counting all of encoder_attn slightly
    OVERcounts the denominator -> this is a conservative (low) estimate of the head's share."""
    head = model.proj_out.weight.numel()  # V*D
    per_step = sum(p.numel() for n, p in model.named_parameters()
                   if n.startswith("model.decoder.layers"))
    return head, per_step, head / (head + per_step)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL, help="HF whisper id (tiny/base/small)")
    ap.add_argument("--clips", type=int, default=12, help="librispeech-dummy clips to transcribe")
    ap.add_argument("--cap", type=int, default=0, help="0 = auto (divisor of V near 41)")
    a = ap.parse_args()

    logger.info(f"loading {a.model} ...")
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).eval()

    W = model.proj_out.weight.detach().float().numpy()  # (V, D)
    V, D = W.shape
    cap = a.cap or pick_cap(V)
    assert V % cap == 0, f"cap {cap} must divide V={V}"
    K = V // cap
    h_n, step_n, share = head_share(model)
    logger.info(f"head proj_out {W.shape}  V={V}=({' * '.join(map(str, _factor(V)))})  "
                f"cap={cap} K={K}")
    logger.info(f"GATE 1 head share of a decode step: {share:.1%}  "
                f"(head {h_n/1e6:.1f}M / +decoder-per-step {step_n/1e6:.1f}M)  "
                f"[Qwen3-0.6B ref 24.8% -> ~1.3x ceiling]")

    import io
    import soundfile as sf
    from datasets import Audio, load_dataset
    # decode=False -> raw flac bytes; decode with soundfile (libsndfile) ourselves, so we don't need
    # datasets' torchcodec/ffmpeg audio backend. Whisper wants 16 kHz mono float (dummy set already is).
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation").cast_column("audio", Audio(decode=False))
    clips = [sf.read(io.BytesIO(ds[i]["audio"]["bytes"]))  # sf.read -> (wav, sr)
             for i in range(min(a.clips, len(ds)))]
    logger.info(f"transcribing {len(clips)} clips for real decoder hidden states ...")
    H, ref = collect_whisper_hidden(model, proc, clips)
    logger.info(f"{H.shape[0]} real decode positions collected")

    logger.info(f"clustering {V} head rows -> K={K} x cap={cap} ...")
    _C, homes = kmeans(W, K, cap)  # kmeans -> (centroids, assignment); build wants the assignment
    Cnorm, Wperm, Vmap = build(W, homes, cap)

    logger.info("GATE 2 agreement vs dense head on real whisper hidden states (recall of the dense "
                "argmax in the flash top-k = task-accuracy retained):")
    for P in (128, 256, 384, 512):
        tk = flash_topk(H, Cnorm, Wperm, Vmap, P, k=5)
        top1 = (flash_topk(H, Cnorm, Wperm, Vmap, P, k=1)[:, 0] == ref).mean()
        top5 = (tk == ref[:, None]).any(1).mean()
        logger.info(f"  P={P:<4d} ({P*cap:>6d}/{V} scored, {P*cap/V:5.1%} vocab)  "
                    f"top-1 {top1:.1%}   top-5 {top5:.1%}")


def _factor(n):
    f, d = [], 2
    while d * d <= n:
        while n % d == 0:
            f.append(d)
            n //= d
        d += 1
    if n > 1:
        f.append(n)
    return f


if __name__ == "__main__":
    main()
