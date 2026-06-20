r"""POC: MULTIPLE (soft) ASSIGNMENT — give each token >1 home cluster to catch the recall tail.

WHY (junior-engineer version)
-----------------------------
Top-1 agreement == stage-1 recall: we hit the dense argmax only when its cluster is among the top-P
probed clusters. The misses live in a heavy tail — tokens whose true cluster ranks ~1000th in the
routing order (p99 required-P ~1165). Re-scoring the clusters (every other routing experiment) can
NEVER fix that: if you don't probe the cluster, no score helps.

Classical IVF answer (Jegou et al., PAMI 2011, "multiple assignment"): stop forcing each token into
exactly one cluster. Put it in its top-r nearest clusters. A tail token whose primary cluster routes
badly often sits near a SECOND cluster that routes easily — give it that second home and you catch it.
A position is now reachable at P if ANY of the token's r homes lands in the top-P.

THE COST, and why it's cheap here
---------------------------------
stage 1 (the DOMINANT head cost: the [K,D] gemv + TopK) is UNCHANGED — same centroids, same K,
same P. Multiple assignment only inflates stage 2: each probed cluster now carries ~r*cap members
instead of cap, so we gather+dot ~r x more candidate rows. Stage 2 is the cheap stage (a Gather and
a small matmul over P*cap rows). So r=2-3 buys recall for a near-free bump on the minor stage.

WHAT THIS POC MEASURES
----------------------
The recall CEILING of soft assignment: keep the shipped centroids and primary partition, give every
token its assigned cluster PLUS its (r-1) next-nearest clusters as extra homes, and recompute
required-P as the MIN rank over the homes. r=1 reproduces the baseline exactly. We report agree@P
across r so you can read the lift at FIXED P (the near-free regime) and also a cost-matched view
(r=2 at P/2 costs ~the same stage-2 as r=1 at P).

Read-only research: imports core helpers, touches nothing in turbohead/.
Run: uv run python experimental/multiple_assignment/multiple_assignment_poc.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import collect_hidden, wikitext

MODEL = "Qwen/Qwen3-0.6B"
HEAD = "artifacts/qwen3_0_6b/head_W.npy"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
MAX_TOKENS = 4000
RS = (1, 2, 3, 4)            # homes per token to sweep
PS = (128, 256, 512)        # probe budgets to report


def homes_for(tokens, W, Cnorm, assigned, r):
    """(U,r) home clusters per token = its assigned cluster + its (r-1) next-nearest by embedding
    cosine. r=1 -> just the assigned cluster (so r=1 == baseline)."""
    St = W[tokens] @ Cnorm                       # (U,K) token-embedding . centroid
    cand = np.argsort(-St, axis=1)[:, :r]        # top-r clusters by embedding match
    has = (cand == assigned[:, None]).any(1)     # is the real cluster already in the top-r?
    homes = cand.copy()
    homes[~has, -1] = assigned[~has]             # if not, force it in (drop the weakest)
    return homes


def main():
    W = np.load(HEAD).astype(np.float32)
    npz = np.load(NPZ)
    Cnorm = npz["Cnorm"].astype(np.float32)      # (D,K)
    Vmap = npz["Vmap"]
    V, D = W.shape
    K, cap = Vmap.shape
    tok2clu = np.empty(V, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=MAX_TOKENS)
    T = H.shape[0]
    sims = H @ Cnorm                             # (T,K) routing scores
    uniq, inv = np.unique(dense, return_inverse=True)   # de-dup the argmax tokens
    assigned_u = tok2clu[uniq]
    print(f"{T} eval positions ({len(uniq)} distinct argmax tokens) | K={K} cap={cap}\n")
    print("required-P = MIN rank over the token's r homes; lower = win.")
    print("stage-2 candidates ~ P*cap*r (stage 1 / gemv unchanged across r).\n")

    rows = {}
    for r in RS:
        homes_u = homes_for(uniq, W, Cnorm, assigned_u, r)      # (U,r)
        homes_pos = homes_u[inv]                                # (T,r) homes per position
        best = np.take_along_axis(sims, homes_pos, axis=1).max(1)   # best-routed home's score
        rank = (sims > best[:, None]).sum(1) + 1
        rows[r] = rank
        pct = {p: int(np.percentile(rank, p)) for p in (50, 90, 99)}
        acc = {P: f"{(rank <= P).mean():.2%}" for P in PS}
        print(f"  r={r}  p50={pct[50]:<3} p90={pct[90]:<4} p99={pct[99]:<6} mean={rank.mean():6.1f} "
              f"| agree @128={acc[128]} @256={acc[256]} @512={acc[512]}")

    # cost-matched: r homes at P/r probed clusters costs ~the same stage-2 as r=1 at P.
    print("\ncost-matched to r=1 @256 (equal stage-2 candidate budget P*cap*r):")
    base = (rows[1] <= 256).mean()
    print(f"  r=1 @256 -> {base:.2%}  (reference)")
    for r in RS[1:]:
        Pm = 256 // r
        print(f"  r={r} @{Pm:<3d} -> {(rows[r] <= Pm).mean():.2%}")


if __name__ == "__main__":
    main()
