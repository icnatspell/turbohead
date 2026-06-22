"""Find the tokens FlashHead most often mis-routes, so the splice can ALWAYS score them and never
miss them (turbohead-splice --always-score).

Why this exists
---------------
FlashHead returns the dense head's top-1 token exactly when that token's cluster is among the top-P
clusters stage 1 picks (stage 2 is exact). A "miss" is when it is not. Measured over real text,
about half the misses are the *same frequent tokens every time* — words/punctuation whose embedding
sits far from its cluster's centroid, so no routing score ever ranks it in. Routing tricks can't fix
a bad cluster assignment, but we can sidestep routing for those few tokens: the spliced graph already
always-scores EOS/BOS via the stage-2 `Wspec`/`spec_ids` path; this script produces an extra list of
ids to add there. A ~64-token list lifts top-1 agreement ~97.5% -> ~98.8% on Qwen3-0.6B at ~free cost.

Output: an .npy of token ids. Feed it to `turbohead-splice --always-score <that.npy>` (omit to disable).
The list is calibration-data- and model-specific — refit per model.
"""

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from turbohead.eval.agreement import wikitext, collect_hidden


def most_missed(dense, true_clu, sims, P, top):
    """Token ids FlashHead misses most often at probe budget `P`, most-frequent first.

    dense    (T,)   the dense head's argmax token at each position (the target).
    true_clu (T,)   the cluster that holds each position's dense argmax token.
    sims     (T,K)  stage-1 routing scores (h . centroids).
    A position is a MISS when its true cluster ranks below `P` in `sims` — i.e. stage 1 would not
    probe it, so stage 2 never scores the winner. We tally the missed `dense` tokens and return the
    `top` most common: the systematic, context-independent misses an always-score list can rescue.
    """
    true_score = sims[np.arange(sims.shape[0]), true_clu]    # routing score of the *right* cluster
    required_p = (sims > true_score[:, None]).sum(1) + 1     # its rank (1 = best); miss if > P
    missed = dense[required_p > P]
    ids, counts = np.unique(missed, return_counts=True)
    return ids[np.argsort(-counts)][:top].astype(np.int64)


def calibrate(model_id, npz_path, P=256, top=64, max_tokens=4000):
    z = np.load(npz_path)
    Cnorm, Vmap = z["Cnorm"].astype(np.float32), z["Vmap"]   # (D,K), (K,cap)
    K, cap = Vmap.shape
    tok2clu = np.empty(K * cap, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)  # token id -> its cluster

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext(max_tokens * 8), max_tokens)
    ids = most_missed(dense, tok2clu[dense], H @ Cnorm, P, top)
    logger.info(f"calibrated {len(ids)} always-score ids over {H.shape[0]} positions "
                f"(P={P}, top={top}); sample {ids[:8].tolist()}")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model (its tokenizer + head)")
    ap.add_argument("--npz", default="artifacts/clusters.npz", help="clusters .npz for this model")
    ap.add_argument("--out", default="artifacts/always_score.npy", help="output token-id .npy")
    ap.add_argument("-P", "--probes", type=int, default=256, help="probe budget to define a 'miss'")
    ap.add_argument("--top", type=int, default=64, help="how many most-missed tokens to keep")
    ap.add_argument("--max-tokens", type=int, default=4000, help="WikiText positions to calibrate on")
    a = ap.parse_args()
    ids = calibrate(a.model, a.npz, a.probes, a.top, a.max_tokens)
    np.save(a.out, ids)
    logger.info(f"saved {a.out}  ({len(ids)} ids) -> turbohead-splice --always-score {a.out}")


if __name__ == "__main__":
    main()
