"""Balanced k-means on head rows -> Cnorm,Wperm,Vmap.
Exact balance K*cap==V (Qwen3: no padding). Constrained Lloyd: unbalanced settle,
then capacity-greedy assignment (anisotropic/MIPS-aware when --eta>1). balanced_assign asserts
exact balance.
Usage: `uv run turbohead-build-clusters [--cap 16 | --clusters K] [--eta E] [--r 2]` (eta 1.0, r 1).
--r 2 = multiple-assignment: each token gets a balanced 2nd home, table cap->2*cap (+~2pp agree@256).

cap = tokens per cluster = FlashHead's cluster ratio (DEFAULT_CLUSTER_RATIO=16); K = V/cap.
cap must divide V exactly (V=151936=2^7*1187 for Qwen3-0.6B -> cap in {1,2,4,8,16,32,64,128,...})."""

import argparse
import numpy as np
from loguru import logger

CAP = 16  # K = V/cap = 9496 for Qwen3-0.6B
# ScaNN anisotropy. 1.0 = plain k-means; the safe default, byte-identical to the historical build.
# eta>1 is a PER-MODEL knob, not a universal win: eta=4 lifts Qwen3-0.6B agreement (+0.2..+1.0pp across
# P) but REGRESSES gemma3-270m (-0.6pp @256). Sweep per model before raising it.
# Rationale + numbers: experimental/anisotropic_clustering/.
ETA = 1.0
CHUNK = 4096


def assign_scores(X, C, cnorm2, eta=1.0, xn=None):
    # argmin ||x-c||^2 == argmax (x·c - .5||c||^2); returns (scores matrix in chunks via caller).
    # eta>1 adds ScaNN's anisotropic penalty (Guo et al., ICML 2020): down-weight a centroid that
    # mismatches x along x's OWN direction (the parallel residual, the only part that moves an inner
    # product). MIPS-aware partition, zero inference cost. eta=1 is byte-identical to plain k-means.
    XC = X @ C.T  # (n, K)
    sc = XC - 0.5 * cnorm2
    if eta != 1.0:  # par = (x - c)·x̂ = ||x|| - c·x̂  ; penalise its square. xn = ||x|| (per chunk).
        par = xn[:, None] - XC / xn[:, None]
        sc = sc - 0.5 * (eta - 1.0) * par ** 2
    return sc


def balanced_assign(X, C, cap, eta=1.0, max_rounds=25, forbid=None):
    """Capacity-greedy: each round, every active token bids its best non-full cluster;
    clusters accept top-bidders up to free slots. Monotonic -> terminates.
    eta>1 scores bids with the anisotropic (MIPS-aware) penalty; eta=1 is plain k-means.
    forbid (V,) bars each token from one cluster (its earlier home) -> a balanced 2nd-home layer
    for multiple-assignment (--r); None = plain single assignment."""
    V, K = X.shape[0], C.shape[0]
    cnorm2 = (C * C).sum(1)
    xn = np.linalg.norm(X, axis=1) + 1e-9  # ||x|| per token, for the anisotropic penalty
    assign = np.full(V, -1, np.int64)
    free = np.full(K, cap, np.int64)
    active = np.arange(V)
    for r in range(max_rounds):
        full_mask = free == 0
        # best non-full cluster + its score for each active token
        best_c = np.empty(active.size, np.int64)
        best_s = np.empty(active.size, np.float32)
        for i in range(0, active.size, CHUNK):
            blk = active[i : i + CHUNK]
            sc = assign_scores(X[blk], C, cnorm2, eta, xn[blk])
            sc[:, full_mask] = -np.inf
            if forbid is not None:
                sc[np.arange(blk.size), forbid[blk]] = -np.inf
            best_c[i : i + CHUNK] = sc.argmax(1)
            best_s[i : i + CHUNK] = sc[np.arange(sc.shape[0]), sc.argmax(1)]
        # per target cluster, accept top free[c] bidders
        accept = np.zeros(active.size, bool)
        order = np.argsort(-best_s)  # global desc by score
        bc = best_c[order]
        taken = free.copy()
        ok = np.zeros(order.size, bool)
        seen = {}
        # ponytail: small python loop over active tokens in score order; O(V) per round, fine for V~150k
        for j, c in enumerate(bc):
            n = seen.get(c, 0)
            if n < taken[c]:
                ok[j] = True
                seen[c] = n + 1
        accept[order[ok]] = True
        acc_tokens = active[accept]
        assign[acc_tokens] = best_c[accept]
        np.subtract.at(free, best_c[accept], 1)
        active = active[~accept]
        logger.info(f"round {r}: assigned {acc_tokens.size}, remaining {active.size}")
        if active.size == 0:
            break
        # ponytail: bail once rounds stall (some heads, e.g. Qwen3-1.7B, drop to ~cap/round and
        # leave most of the vocab — the block tail below places the remainder far faster).
        if acc_tokens.size <= 4 * cap:
            break
    # tail: greedy nearest-free fill of the remainder. Score a BLOCK of tokens against all centroids
    # in one matmul (then mask full clusters) instead of a per-token gemv that re-copies C[free_c]
    # every iteration — the remainder can be most of the vocab when the rounds stall.
    if active.size:
        logger.info(f"sequential finish for {active.size} tokens")
        for i in range(0, active.size, CHUNK):
            blk = active[i : i + CHUNK]
            sc = assign_scores(X[blk], C, cnorm2, eta, xn[blk])  # (blk, K)
            sc[:, free == 0] = -np.inf
            if forbid is not None:
                sc[np.arange(blk.size), forbid[blk]] = -np.inf
            for j, tok in enumerate(blk):
                c = int(sc[j].argmax())
                assign[tok] = c
                free[c] -= 1
                if free[c] == 0:
                    sc[:, c] = -np.inf  # close cluster for the rest of this block
    assert (free == 0).all() and np.bincount(assign, minlength=K).min() == cap
    return assign


def centroids_from(W, assign, K, cap):
    order = np.argsort(assign, kind="stable")
    return W[order].reshape(K, cap, W.shape[1]).mean(1)  # exact cap each


def kmeans(W, K, cap, eta=1.0, settle_iters=15, balanced_iters=5):
    V, D = W.shape
    rng = np.random.default_rng(0)
    C = W[rng.choice(V, K, replace=False)].copy()
    for it in range(settle_iters):  # unbalanced Lloyd settles good centroids fast (isotropic warm start)
        cnorm2 = (C * C).sum(1)
        a = np.empty(V, np.int64)
        for i in range(0, V, CHUNK):
            a[i : i + CHUNK] = assign_scores(W[i : i + CHUNK], C, cnorm2).argmax(1)
        newC = np.zeros_like(C)
        np.add.at(newC, a, W)
        cnt = np.bincount(a, minlength=K)[:, None]
        nz = cnt[:, 0] > 0
        newC[nz] /= cnt[nz]
        shift = np.linalg.norm(newC - C)
        C = newC
        logger.info(f"settle iter {it}: shift {shift:.3f}")
    # constrained Lloyd: balanced (anisotropic if eta>1) assign <-> mean centroid from balanced members.
    # The member mean is kept on purpose: ScaNN's closed-form centroid was tested and lost (cosine
    # routing discards the magnitude it tunes). See experimental/anisotropic_clustering/.
    assign = balanced_assign(W, C, cap, eta)
    for it in range(balanced_iters):
        C = centroids_from(W, assign, K, cap)
        assign = balanced_assign(W, C, cap, eta)
        logger.info(f"balanced iter {it} done")
    return C, assign


def build(W, assign, cap):
    """assign: (V,) single home, or (V,r) home layers (col 0 = primary) for multiple-assignment.
    Each layer packs cap tokens/cluster; the table is (K, r*cap) so the fused op (reads cap from the
    table shape) gathers r*cap candidates/cluster. Cnorm stays the PRIMARY (sharp) centroids — never
    blurred by the 2nd-home layers (rebuilding over doubled members loses; see experimental/)."""
    V, D = W.shape
    K = V // cap
    homes = assign if assign.ndim == 2 else assign[:, None]  # (V, r)
    vb, wb = [], []
    for j in range(homes.shape[1]):
        order = np.argsort(homes[:, j], kind="stable")  # groups by cluster, cap each
        vb.append(order.reshape(K, cap))
        wb.append(W[order].reshape(K, cap, D))
    Vmap = np.concatenate(vb, axis=1).astype(np.int64)        # (K, r*cap)
    Wperm = np.concatenate(wb, axis=1).astype(np.float16)     # (K, r*cap, D), stored fp16
    centroids = np.stack([W[vb[0][k]].mean(0) for k in range(K)])  # PRIMARY members only
    cn = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    Cnorm = cn.T.astype(np.float16)  # (D,K) columns
    return Cnorm, Wperm, Vmap


def main():
    ap = argparse.ArgumentParser(description="Balanced k-means over head rows -> clusters.npz")
    ap.add_argument("--head", default="artifacts/head_W.npy")
    ap.add_argument("--out", default="artifacts/clusters.npz")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cap", type=int, default=CAP,
                   help="tokens per cluster = cluster ratio (default 16). K = V/cap")
    g.add_argument("--clusters", type=int, help="number of clusters K; sets cap = V/K")
    ap.add_argument("--eta", type=float, default=ETA,
                    help="ScaNN anisotropy (default 1.0 = plain k-means): parallel-error weight in the "
                         "partition. eta>1 can lift top-1 agreement at zero inference cost, but it is "
                         "PER-MODEL (helps Qwen3-0.6B, hurts gemma3-270m). Sweep before raising it.")
    ap.add_argument("--r", type=int, default=1, choices=(1, 2),
                    help="homes per token (multiple-assignment, default 1). r=2 gives each token a "
                         "balanced 2nd home (table grows cap->2*cap), lifting top-1 agreement ~+2pp@256 "
                         "by catching the recall tail. Costs stage-2 gather; pair with int8 "
                         "(turbohead-splice --head-weight-dtype int8) to keep it ~speed-neutral.")
    a = ap.parse_args()

    W = np.load(a.head).astype(np.float32)
    V, D = W.shape
    cap = V // a.clusters if a.clusters else a.cap
    K = V // cap
    knob = f"clusters={a.clusters}" if a.clusters else f"cap={cap}"
    assert K * cap == V, f"V={V} not divisible by {knob} (cap={cap}); pick a divisor of {V}"
    logger.info(f"V={V} D={D} K={K} cap={cap} eta={a.eta} r={a.r}")
    C, primary = kmeans(W, K, cap, a.eta)
    # extra balanced home layers: layer j bars the token's previous home (ponytail: single forbid, so
    # r capped at 2 — the validated sweet spot; r>=3 loses cost-matched and would need a forbid-set).
    homes = primary
    if a.r > 1:
        layers = [primary]
        for _ in range(1, a.r):
            layers.append(balanced_assign(W, C, cap, a.eta, forbid=layers[-1]))
        homes = np.stack(layers, axis=1)
    Cnorm, Wperm, Vmap = build(W, homes, cap)
    np.savez(a.out, Cnorm=Cnorm, Wperm=Wperm, Vmap=Vmap)
    logger.info(f"saved {a.out}  Cnorm{Cnorm.shape} Wperm{Wperm.shape} Vmap{Vmap.shape}")
    logger.info("run `uv run turbohead-agreement` for real-hidden-state agreement")


if __name__ == "__main__":
    main()
