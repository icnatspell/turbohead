r"""POC: ScaNN-style ANISOTROPIC clustering — does a MIPS-aware partition lift top-1 agreement?

WHY (read this first, junior-engineer version)
----------------------------------------------
Top-1 agreement == stage-1 recall: FlashHead returns the dense argmax token exactly when that
token's cluster lands in the top-P probed clusters. So agreement is decided entirely by HOW WE
PARTITION the vocabulary into clusters and HOW WE SCORE those clusters at run time.

Every earlier experiment (mips_routing, dataaware_routing, recall_lift) froze the partition
`build_clusters.py` produces and only re-scored it. But that partition is built by plain k-means:
it minimises  ||e - c||^2 , the Euclidean error between a token embedding `e` and its centroid `c`.

Here's the mismatch ScaNN (Guo et al., ICML 2020, "Anisotropic Vector Quantization") points out:
we do not search by Euclidean distance. We search by INNER PRODUCT — we want the centroid `c` to
predict  <h, e>  (hidden state dotted with the token's embedding), not to sit physically near `e`.
And for an inner product, the part of the quantization error that actually changes the score is the
component PARALLEL to `e` (along the direction queries arrive from); the orthogonal part barely
moves <h, c>. Ordinary k-means weights both equally and so "wastes" centroid accuracy on the
direction that doesn't matter for ranking.

ScaNN's fix: weight the parallel error more. Assign token `x` to the centroid `c` minimising

    L(x, c) = ||x - c||^2  +  (eta - 1) * ( (x - c) . x_hat )^2          # x_hat = x / ||x||
              \___isotropic__/   \___extra penalty on the PARALLEL residual___/

eta = 1 recovers plain k-means (sanity check). eta > 1 pulls clusters into a shape that preserves
inner-product RANKING, which is exactly what stage-1 recall needs. Cost at inference: ZERO — same
(D,K) centroid matrix, same gemv. Only `build_clusters.py` changes, offline.

WHAT THIS POC MEASURES
----------------------
We rebuild the balanced partition with the anisotropic assignment rule for a few values of `eta`,
then measure `required_p` (rank of each position's true cluster in the cosine routing order) on
REAL hidden states, exactly like the other POCs. Lower required_p / higher agree@P = win.

CENTROID UPDATE: two variants, both run here so you can see what the centroid choice is worth.
  - MEAN: the plain member mean (the isotropic optimum). The original lower bound: anisotropic
    GROUPING but an unaltered centroid.
  - SCANN: the closed-form parallel-weighted centroid (centroids_scann). A per-cluster weighted
    least-squares solve, reduced via Woodbury to a cap×cap system. This is the full method.

Read-only research: imports core helpers but touches nothing in turbohead/.
Run: uv run python experimental/anisotropic_clustering/anisotropic_clustering_poc.py [eta ...]
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import collect_hidden, wikitext

MODEL = "Qwen/Qwen3-0.6B"
HEAD = "artifacts/qwen3_0_6b/head_W.npy"        # (V,D) raw lm_head rows = token embeddings
NPZ = "artifacts/qwen3_0_6b/clusters.npz"       # the shipped baseline partition
CHUNK = 4096
ITERS = 3                                        # balanced refinement steps from the warm start
MAX_TOKENS = 4000                                # eval hidden states
SEED = 0


def aniso_scores(X, C, eta):
    """Higher = better (we maximise -L). For eta=1 this is exactly the k-means assignment score
    argmax(x.c - .5||c||^2), so the baseline is reproduced. Derivation, dropping the per-token
    constant ||x||^2:  -L = 2 x.c - ||c||^2 - (eta-1)(||x|| - c.x_hat)^2 ."""
    XC = X @ C.T                                  # (n,K)  x . c
    cn2 = (C * C).sum(1)                          # (K,)   ||c||^2
    xn = np.linalg.norm(X, axis=1) + 1e-9         # (n,)   ||x||
    par = xn[:, None] - XC / xn[:, None]          # (n,K)  (x - c).x_hat  =  ||x|| - c.x_hat
    return 2 * XC - cn2[None, :] - (eta - 1.0) * par ** 2


def balanced_assign(X, C, cap, eta, max_rounds=25):
    """Capacity-greedy balanced assignment (same skeleton as surgery/build_clusters.py) but with the
    anisotropic score. Each round every still-unassigned token bids for its best non-full cluster;
    clusters take the top bidders up to free slots. A block tail fills any stalled remainder."""
    V, K = X.shape[0], C.shape[0]
    assign = np.full(V, -1, np.int64)
    free = np.full(K, cap, np.int64)
    active = np.arange(V)
    for r in range(max_rounds):
        full_mask = free == 0
        best_c = np.empty(active.size, np.int64)
        best_s = np.empty(active.size, np.float32)
        for i in range(0, active.size, CHUNK):
            sc = aniso_scores(X[active[i:i + CHUNK]], C, eta)
            sc[:, full_mask] = -np.inf
            best_c[i:i + CHUNK] = sc.argmax(1)
            best_s[i:i + CHUNK] = sc[np.arange(sc.shape[0]), sc.argmax(1)]
        order = np.argsort(-best_s)               # global desc by bid score
        ok = np.zeros(order.size, bool)
        seen = {}
        for j, c in enumerate(best_c[order]):     # accept top free[c] bidders per cluster
            n = seen.get(c, 0)
            if n < free[c]:
                ok[j] = True
                seen[c] = n + 1
        accept = np.zeros(active.size, bool)
        accept[order[ok]] = True
        n_accept = int(accept.sum())
        assign[active[accept]] = best_c[accept]
        np.subtract.at(free, best_c[accept], 1)
        active = active[~accept]
        if active.size == 0 or n_accept <= 4 * cap:   # done, or rounds stalled -> block tail
            break
    if active.size:                               # nearest-free greedy fill of the remainder
        for i in range(0, active.size, CHUNK):
            blk = active[i:i + CHUNK]
            sc = aniso_scores(X[blk], C, eta)
            sc[:, free == 0] = -np.inf
            for j, tok in enumerate(blk):
                c = int(sc[j].argmax())
                assign[tok] = c
                free[c] -= 1
                if free[c] == 0:
                    sc[:, c] = -np.inf
    assert (free == 0).all()
    return assign


def centroids_mean(W, assign, K, cap):
    order = np.argsort(assign, kind="stable")
    return W[order].reshape(K, cap, W.shape[1]).mean(1)   # member mean (isotropic optimum)


def centroids_scann(W, assign, K, cap, eta):
    """ScaNN's closed-form parallel-weighted centroid (the bit the member-mean PoC skipped).

    For one cluster with members x_i, minimise  Σ_i ||x_i-c||² + (eta-1)((x_i-c).x̂_i)² .
    Setting the gradient to 0 gives  A c = b  with  A = Σ_i (I + (eta-1) x̂_i x̂_iᵀ),
    b = Σ_i (I + (eta-1) x̂_i x̂_iᵀ) x_i = eta Σ_i x_i  (because x̂_i.x_i = ||x_i||).

    A is D×D, but it is nI plus a rank-(cap) update  (eta-1) X̂ᵀX̂, so Woodbury turns the solve into
    one cap×cap system per cluster (cap=16 here, not 1024). eta=1 is singular (no parallel penalty) →
    fall back to the member mean, which is what A c = b reduces to there anyway."""
    if eta == 1.0:
        return centroids_mean(W, assign, K, cap)
    order = np.argsort(assign, kind="stable")
    Xb = W[order].reshape(K, cap, W.shape[1])             # (K,cap,D) cluster members
    Xhat = Xb / (np.linalg.norm(Xb, axis=2, keepdims=True) + 1e-9)
    b = eta * Xb.sum(1)                                   # (K,D)
    G = np.einsum("kcd,ked->kce", Xhat, Xhat)             # (K,cap,cap)  X̂ X̂ᵀ
    Xhb = np.einsum("kcd,kd->kc", Xhat, b)                # (K,cap)      X̂ b
    inner = G / cap + np.eye(cap) / (eta - 1.0)           # (K,cap,cap)  D̃⁻¹ + (1/n)G
    y = np.linalg.solve(inner, Xhb[..., None])[..., 0]    # (K,cap)  per-cluster cap×cap solve
    return b / cap - np.einsum("kcd,kc->kd", Xhat, y) / cap ** 2


def cluster(W, C0, cap, eta, iters, scann=True):
    """Warm-start from the shipped centroids, then alternate (anisotropic assign) / (centroid update).
    scann=True uses ScaNN's closed-form centroid; False uses the member mean (the old lower bound).
    Returns (assign[V] = token->cluster, Cnorm[K,D] = unit routing directions)."""
    K = C0.shape[0]
    C = C0.copy()
    upd = (lambda a: centroids_scann(W, a, K, cap, eta)) if scann else (lambda a: centroids_mean(W, a, K, cap))
    for it in range(iters):
        assign = balanced_assign(W, C, cap, eta)
        C = upd(assign)
        print(f"    eta={eta} iter {it} done")
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return assign, Cn


def required_p(sims, true_clu):
    ts = sims[np.arange(sims.shape[0]), true_clu]
    return (sims > ts[:, None]).sum(1) + 1


def report(name, rank):
    pct = {p: int(np.percentile(rank, p)) for p in (50, 90, 99)}
    acc = {P: f"{(rank <= P).mean():.2%}" for P in (128, 256, 512)}
    print(f"  {name:16s} p50={pct[50]:<3} p90={pct[90]:<4} p99={pct[99]:<6} "
          f"mean={rank.mean():7.1f} | agree @128={acc[128]} @256={acc[256]} @512={acc[512]}")


def main():
    etas = [float(a) for a in sys.argv[1:]] or [2.0, 6.0]
    W = np.load(HEAD).astype(np.float32)          # (V,D)
    npz = np.load(NPZ)
    Vmap = npz["Vmap"]
    V, D = W.shape
    K, cap = Vmap.shape
    mu0 = npz["Wperm"].astype(np.float32).mean(1)   # warm-start centroids = shipped cluster means

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=MAX_TOKENS)
    print(f"{H.shape[0]} eval positions | V={V} D={D} K={K} cap={cap} | iters={ITERS}\n")

    # baseline: the shipped partition, cosine routing on its mean centroids
    print("BASELINE  shipped k-means partition (eta=1, isotropic):")
    tok2clu = np.empty(V, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)
    report("baseline", required_p(H @ npz["Cnorm"].astype(np.float32), tok2clu[dense]))

    # anisotropic partitions: mean centroid (old lower bound) vs ScaNN closed form (the finish)
    print("\nANISOTROPIC  re-clustered with parallel-error penalty (lower required_p = win):")
    for eta in etas:
        assign, Cn = cluster(W, mu0, cap, eta, ITERS, scann=False)   # member-mean centroid
        report(f"eta={eta} mean", required_p(H @ Cn.T, assign[dense]))
        assign, Cn = cluster(W, mu0, cap, eta, ITERS, scann=True)    # ScaNN closed-form centroid
        report(f"eta={eta} scann", required_p(H @ Cn.T, assign[dense]))


if __name__ == "__main__":
    main()
