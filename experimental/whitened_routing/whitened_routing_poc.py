r"""POC: WHITENED (Mahalanobis) ROUTING — a learned linear metric folded into the centroids, free.

WHY (junior-engineer version)
-----------------------------
Stage-1 routing scores clusters by cosine, h . c_hat. But LLM hidden states are ANISOTROPIC: a few
dimensions have huge variance and dominate every dot product, so the routing order is driven by those
dims whether or not they carry the signal that picks the right cluster. The classical fix from
nearest-neighbour search is a Mahalanobis metric: score with  h^T A c  for a learned matrix A that
de-emphasises the high-variance directions (A = inverse covariance of the hidden states "whitens"
the space).

The trick that makes it FREE: an inner product with a fixed metric folds entirely into the stored
centroid. h^T A c = h . (A c). Precompute c' = A c offline; at run time you do the SAME h . c' gemv
over the SAME (D,K) matrix. Zero extra ops. (A pure ROTATION would be a no-op here -- R preserves
inner products -- which is why OPQ-style rotation only helps once you add a product quantizer, the
inverted_multi_index experiment. The non-trivial free lever for a single codebook is this whitening.)

recall_lift already tried DIAGONAL whitening (scale each dim by 1/std) and mean-centering, with a
small gain. This is the full-rank generalisation: the whole inverse-covariance matrix, which also
removes cross-dimension correlations a diagonal can't.

WHAT THIS POC MEASURES
----------------------
Fit mu and Sigma on a train split of hidden states, fold A=(Sigma+lambda I)^-1 into the shipped
centroids, and measure held-out required-P. Sweep the shrinkage lambda (Sigma is noisy at D=1024).
Compare against cosine baseline and the diagonal-whitening reference. Lower required-P = win, and
it costs nothing at inference.

Read-only research: imports core helpers, touches nothing in turbohead/.
Run: uv run python experimental/whitened_routing/whitened_routing_poc.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import collect_hidden, wikitext

MODEL = "Qwen/Qwen3-0.6B"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
MAX_TOKENS = 8000
PS = (128, 256, 512)
SEED = 0


def required_p(sims, true_clu):
    ts = sims[np.arange(sims.shape[0]), true_clu]
    return (sims > ts[:, None]).sum(1) + 1


def report(name, rank, note):
    pct = {p: int(np.percentile(rank, p)) for p in (50, 90, 99)}
    acc = {P: f"{(rank <= P).mean():.2%}" for P in PS}
    print(f"  {name:20s} p50={pct[50]:<3} p90={pct[90]:<4} p99={pct[99]:<6} mean={rank.mean():6.1f} "
          f"| @128={acc[128]} @256={acc[256]} @512={acc[512]}  [{note}]")


def main():
    rng = np.random.default_rng(SEED)
    npz = np.load(NPZ)
    Cnorm = npz["Cnorm"].astype(np.float32)          # (D,K) unit centroid directions
    Vmap = npz["Vmap"]
    D, K = Cnorm.shape
    cap = Vmap.shape[1]
    tok2clu = np.empty(K * cap, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext(80000), max_tokens=MAX_TOKENS)
    true_clu = tok2clu[dense]
    T = H.shape[0]
    perm = rng.permutation(T)
    tr, te = perm[:T // 2], perm[T // 2:]
    Hte, cte = H[te], true_clu[te]
    print(f"{T} positions ({len(tr)} fit / {len(te)} eval) | K={K} cap={cap} D={D}\n")

    report("cosine (baseline)", required_p(Hte @ Cnorm, cte), "current")

    mu = H[tr].mean(0)
    Hc = H[tr] - mu
    Sigma = (Hc.T @ Hc) / len(tr)                    # (D,D) covariance of train hidden states
    tau = np.trace(Sigma) / D                        # mean variance, for shrinkage scale

    # diagonal whitening reference (what recall_lift tried): A = diag(1/var)
    Adiag = 1.0 / (np.diag(Sigma) + 1e-6)
    cdiag = Cnorm * Adiag[:, None]                    # fold diag metric into centroids
    report("diag-whiten (ref)", required_p((Hte - mu) @ cdiag, cte), "free; folds in")

    # full-rank Mahalanobis: A = (Sigma + lambda*tau*I)^-1, shrinkage lambda swept
    for lam in (1.0, 0.1, 0.03, 0.01):
        A = np.linalg.inv(Sigma + lam * tau * np.eye(D, dtype=np.float32))
        cprime = A @ Cnorm                           # (D,K) folded centroids
        report(f"mahalanobis lam={lam}", required_p((Hte - mu) @ cprime, cte), "free; folds in")
        cn = cprime / (np.linalg.norm(cprime, axis=0, keepdims=True) + 1e-9)   # re-normalised cols
        report(f"  + col-norm lam={lam}", required_p((Hte - mu) @ cn, cte), "free; folds in")


if __name__ == "__main__":
    main()
