r"""POC: do anisotropic_clustering (#1) and multiple_assignment (#2) STACK?

They attack different things: #1 reshapes the PARTITION (MIPS-aware clusters), #2 adds HOMES to
whatever partition exists. Orthogonal in principle, so the gains should be roughly additive — but
"should" isn't "does", e.g. a better partition might already capture some of the tail that the second
home was rescuing, leaving less for #2 to add.

Clean 2x2, one consistent pipeline (warm-start + 3 balanced passes) so the only thing that varies is
eta and r, and the cross-term is isolated:

    (eta=1, r=1)  control
    (eta=4, r=1)  anisotropic alone
    (eta=1, r=2)  multiple-assignment alone
    (eta=4, r=2)  stacked

We reuse the real functions from both experiments (this is a stack TEST of them, so importing them is
the point — no re-derivation). Metric: required_p = min rank over homes, on real hidden states.

Run: uv run python experimental/combinations/aniso_x_multiassign_poc.py
"""
import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import collect_hidden, wikitext

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "anisotropic_clustering"))
sys.path.insert(0, os.path.join(HERE, "..", "multiple_assignment"))
from anisotropic_clustering_poc import cluster          # noqa: E402  (assign, Cn[K,D])
from multiple_assignment_poc import homes_for           # noqa: E402  (U,r) home clusters

MODEL = "Qwen/Qwen3-0.6B"
HEAD = "artifacts/qwen3_0_6b/head_W.npy"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
MAX_TOKENS = 4000
ITERS = 3
PS = (128, 256, 512)


def recall(H, W, Cn, assign, dense, r):
    """required_p = min rank over the token's r homes, under this partition's cosine routing."""
    sims = H @ Cn.T                                       # (T,K)
    uniq, inv = np.unique(dense, return_inverse=True)
    homes_u = homes_for(uniq, W, Cn.T, assign[uniq], r)  # (U,r)
    best = np.take_along_axis(sims, homes_u[inv], axis=1).max(1)
    return (sims > best[:, None]).sum(1) + 1


def line(name, rank):
    pct = {p: int(np.percentile(rank, p)) for p in (50, 90, 99)}
    acc = {P: f"{(rank <= P).mean():.2%}" for P in PS}
    print(f"  {name:22s} p50={pct[50]:<3} p90={pct[90]:<4} p99={pct[99]:<5} mean={rank.mean():6.1f} "
          f"| @128={acc[128]} @256={acc[256]} @512={acc[512]}")


def main():
    W = np.load(HEAD).astype(np.float32)
    npz = np.load(NPZ)
    mu0 = npz["Wperm"].astype(np.float32).mean(1)        # warm-start centroids
    K, cap = npz["Vmap"].shape

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=MAX_TOKENS)
    print(f"{H.shape[0]} positions | K={K} cap={cap} | iters={ITERS}\n")

    parts = {}
    for eta in (1.0, 4.0):
        assign, Cn = cluster(W, mu0, cap, eta, ITERS)     # build the partition once per eta
        parts[eta] = (assign, Cn)
        print(f"  built partition eta={eta}")
    print()

    a256 = {}
    for eta in (1.0, 4.0):
        assign, Cn = parts[eta]
        for r in (1, 2):
            rank = recall(H, W, Cn, assign, dense, r)
            line(f"eta={eta} r={r}", rank)
            a256[(eta, r)] = (rank <= 256).mean()

    # is the joint gain >= the sum of the two solo gains? (additivity check on agree@256)
    base = a256[(1.0, 1)]
    g_aniso = a256[(4.0, 1)] - base
    g_multi = a256[(1.0, 2)] - base
    g_joint = a256[(4.0, 2)] - base
    print(f"\nagree@256 deltas vs control: aniso +{g_aniso:.2%}  multi +{g_multi:.2%}  "
          f"joint +{g_joint:.2%}  (sum of solos +{g_aniso + g_multi:.2%})")
    print(f"-> stacks {'fully/super-additively' if g_joint >= g_aniso + g_multi - 1e-9 else 'partially (sub-additive)'}")


if __name__ == "__main__":
    main()
