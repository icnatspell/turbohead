"""Phase 1 — balanced k-means on head rows -> Cnorm,Wperm,Vmap (§3).
Exact balance K*cap==V (Qwen3: no padding). Constrained Lloyd: unbalanced settle,
then capacity-greedy assignment. balanced_assign asserts exact balance.
Usage: `uv run turbohead-build-clusters [--cap 16 | --clusters K]`.

cap = tokens per cluster = FlashHead's cluster ratio (DEFAULT_CLUSTER_RATIO=16); K = V/cap.
cap must divide V exactly (V=151936=2^7*1187 for Qwen3-0.6B -> cap in {1,2,4,8,16,32,64,128,...})."""

import argparse
import numpy as np
from loguru import logger

CAP = 16  # K = V/cap = 9496 for Qwen3-0.6B
CHUNK = 4096


def assign_scores(X, C, cnorm2):
    # argmin ||x-c||^2 == argmax (x·c - .5||c||^2); returns (scores matrix in chunks via caller)
    return X @ C.T - 0.5 * cnorm2  # (n, K)


def balanced_assign(X, C, cap, max_rounds=25):
    """Capacity-greedy: each round, every active token bids its best non-full cluster;
    clusters accept top-bidders up to free slots. Monotonic -> terminates."""
    V, K = X.shape[0], C.shape[0]
    cnorm2 = (C * C).sum(1)
    assign = np.full(V, -1, np.int64)
    free = np.full(K, cap, np.int64)
    active = np.arange(V)
    for r in range(max_rounds):
        full_mask = free == 0
        # best non-full cluster + its score for each active token
        best_c = np.empty(active.size, np.int64)
        best_s = np.empty(active.size, np.float32)
        for i in range(0, active.size, CHUNK):
            sc = assign_scores(X[active[i : i + CHUNK]], C, cnorm2)
            sc[:, full_mask] = -np.inf
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
    # tail: sequential greedy fill (small remainder; rounds stall here otherwise)
    if active.size:
        logger.info(f"sequential finish for {active.size} tokens")
        free_c = np.where(free > 0)[0]
        for tok in active:
            sc = X[tok] @ C[free_c].T - 0.5 * cnorm2[free_c]
            c = free_c[sc.argmax()]
            assign[tok] = c
            free[c] -= 1
            if free[c] == 0:
                free_c = np.where(free > 0)[0]
    assert (free == 0).all() and np.bincount(assign, minlength=K).min() == cap
    return assign


def centroids_from(W, assign, K, cap):
    order = np.argsort(assign, kind="stable")
    return W[order].reshape(K, cap, W.shape[1]).mean(1)  # exact cap each


def kmeans(W, K, cap, settle_iters=15, balanced_iters=5):
    V, D = W.shape
    rng = np.random.default_rng(0)
    C = W[rng.choice(V, K, replace=False)].copy()
    for it in range(settle_iters):  # unbalanced Lloyd settles good centroids fast
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
    # constrained Lloyd: balanced assign <-> recompute centroids from balanced members
    assign = balanced_assign(W, C, cap)
    for it in range(balanced_iters):
        C = centroids_from(W, assign, K, cap)
        assign = balanced_assign(W, C, cap)
        logger.info(f"balanced iter {it} done")
    return C, assign


def build(W, assign, cap):
    V, D = W.shape
    K = V // cap
    # tokens of each cluster, packed in cluster order
    order = np.argsort(assign, kind="stable")  # groups by cluster, cap each
    Vmap = order.reshape(K, cap).astype(np.int64)
    Wperm = W[order].reshape(K, cap, D).astype(np.float16)  # §7 v1: fp16
    centroids = np.stack([W[Vmap[k]].mean(0) for k in range(K)])  # recompute from final members
    cn = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    Cnorm = cn.T.astype(np.float16)  # (D,K) columns, §3
    return Cnorm, Wperm, Vmap


def main():
    ap = argparse.ArgumentParser(description="Balanced k-means over head rows -> clusters.npz")
    ap.add_argument("--head", default="artifacts/head_W.npy")
    ap.add_argument("--out", default="artifacts/clusters.npz")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cap", type=int, default=CAP,
                   help="tokens per cluster = cluster ratio (default 16). K = V/cap")
    g.add_argument("--clusters", type=int, help="number of clusters K; sets cap = V/K")
    a = ap.parse_args()

    W = np.load(a.head).astype(np.float32)
    V, D = W.shape
    cap = V // a.clusters if a.clusters else a.cap
    K = V // cap
    knob = f"clusters={a.clusters}" if a.clusters else f"cap={cap}"
    assert K * cap == V, f"V={V} not divisible by {knob} (cap={cap}); pick a divisor of {V}"
    logger.info(f"V={V} D={D} K={K} cap={cap}")
    C, assign = kmeans(W, K, cap)
    Cnorm, Wperm, Vmap = build(W, assign, cap)
    np.savez(a.out, Cnorm=Cnorm, Wperm=Wperm, Vmap=Vmap)
    logger.info(f"saved {a.out}  Cnorm{Cnorm.shape} Wperm{Wperm.shape} Vmap{Vmap.shape}")
    logger.info("run `uv run turbohead-agreement` for real-hidden-state agreement")


if __name__ == "__main__":
    main()
