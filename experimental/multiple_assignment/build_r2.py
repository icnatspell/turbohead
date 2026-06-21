r"""Real balanced r=2 build + agreement check (the graduation step for multiple_assignment).

GRADUATED (2026-06-21): this is now in core as `turbohead-build-clusters --r 2` (balanced 2nd home via
build_clusters.balanced_assign(forbid=...), table built by build()). Sampling dedup lives in
decode_loop._pick. Kept here as the standalone evidence + the int8-vs-fp32 agreement A/B; the core
command re-clusters from scratch, this script reuses the shipped clusters.npz primary.

The PoC (multiple_assignment_poc.py) measured the recall CEILING with each token's true 2nd-nearest
centroid (unbalanced — some clusters get many seconds, some none). The fused op needs a FIXED-width
table: it reads cap from Wperm.Shape()[1] and loops cap rows per probed cluster. So the real build
must give each cluster EXACTLY cap second-homes -> a balanced 2nd assignment, table grows cap -> 2cap.

This script:
  1. keeps the SHARP shipped centroids + primary partition (clusters.npz); does NOT re-cluster
     (rebuilding over doubled memberships blurs them and loses — see cost_matched_real_build_poc.py).
  2. runs a balanced 2nd assignment over the same sharp centroids, excluding each token's primary
     cluster, so every cluster receives exactly cap seconds.
  3. packs clusters_r2.npz: Wperm (K,2cap,D), Vmap (K,2cap), Cnorm unchanged.
  4. measures top-1 agreement vs dense at P=128/256, fp32 AND int8-roundtrip weights, to answer two
     questions: does balancing the seconds keep the PoC's +2.25pp, and does int8 quant survive it.

Read-only on core; vendors a balanced_assign variant (core's has no per-token exclusion).
Run: uv run python experimental/multiple_assignment/build_r2.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import collect_hidden, wikitext, flash_top1

MODEL = "Qwen/Qwen3-0.6B"
HEAD = "artifacts/qwen3_0_6b/head_W.npy"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
OUT = "artifacts/qwen3_0_6b/clusters_r2.npz"
CHUNK = 4096


def balanced_assign_excl(X, C, cap, forbid, max_rounds=25):
    """Capacity-greedy balanced assignment (core's balanced_assign), but each token may NOT pick the
    cluster in forbid[token] (its primary home). Result: exactly cap tokens per cluster, none on its
    own primary -> a clean 2nd-home layer."""
    V, K = X.shape[0], C.shape[0]
    cnorm2 = (C * C).sum(1)
    assign = np.full(V, -1, np.int64)
    free = np.full(K, cap, np.int64)
    active = np.arange(V)
    for rnd in range(max_rounds):
        full_mask = free == 0
        best_c = np.empty(active.size, np.int64)
        best_s = np.empty(active.size, np.float32)
        for i in range(0, active.size, CHUNK):
            blk = active[i : i + CHUNK]
            sc = X[blk] @ C.T - 0.5 * cnorm2
            sc[:, full_mask] = -np.inf
            sc[np.arange(blk.size), forbid[blk]] = -np.inf  # can't re-home in the primary
            best_c[i : i + CHUNK] = sc.argmax(1)
            best_s[i : i + CHUNK] = sc[np.arange(sc.shape[0]), sc.argmax(1)]
        order = np.argsort(-best_s)
        bc = best_c[order]
        taken = free.copy()
        ok = np.zeros(order.size, bool)
        seen = {}
        for j, c in enumerate(bc):
            n = seen.get(c, 0)
            if n < taken[c]:
                ok[j] = True
                seen[c] = n + 1
        accept = np.zeros(active.size, bool)
        accept[order[ok]] = True
        assign[active[accept]] = best_c[accept]
        np.subtract.at(free, best_c[accept], 1)
        active = active[~accept]
        print(f"  2nd-assign round {rnd}: placed {accept.sum()}, remaining {active.size}")
        if active.size == 0:
            break
        if accept.sum() <= 4 * cap:
            break
    # tail: greedy nearest-free fill, still excluding the primary
    if active.size:
        print(f"  sequential finish for {active.size} tokens")
        for i in range(0, active.size, CHUNK):
            blk = active[i : i + CHUNK]
            sc = X[blk] @ C.T - 0.5 * cnorm2
            sc[:, free == 0] = -np.inf
            sc[np.arange(blk.size), forbid[blk]] = -np.inf
            for j, tok in enumerate(blk):
                c = int(sc[j].argmax())
                assign[tok] = c
                free[c] -= 1
                if free[c] == 0:
                    sc[:, c] = -np.inf
    assert (free == 0).all() and np.bincount(assign, minlength=K).min() == cap
    assert (assign != forbid).all(), "a token re-homed in its primary"
    return assign


def int8_roundtrip(Wp):
    """Per-row symmetric int8 quant -> dequant, the fused_q8 path (docs/FUSED_HEAD_INT8.md)."""
    flat = Wp.reshape(-1, Wp.shape[-1])
    scale = np.abs(flat).max(1, keepdims=True) / 127.0
    q = np.round(flat / scale).clip(-127, 127).astype(np.int8)
    return (q.astype(np.float32) * scale).reshape(Wp.shape)


def main():
    W = np.load(HEAD).astype(np.float32)
    npz = np.load(NPZ)
    Cnorm = npz["Cnorm"].astype(np.float32)   # (D,K) sharp, unchanged
    Vmap = npz["Vmap"]                         # (K,cap) primary
    V, D = W.shape
    K, cap = Vmap.shape

    # sharp member-mean centroids (what build() normalized into Cnorm); primary cluster per token
    C = W[Vmap.reshape(-1)].reshape(K, cap, D).mean(1)   # (K,D)
    primary = np.empty(V, np.int64)
    primary[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)

    print(f"V={V} D={D} K={K} cap={cap} -> r=2 table (K,{2*cap},D)")
    second = balanced_assign_excl(W, C, cap, primary)   # (V,) 2nd home, balanced, != primary

    # pack the second layer per cluster, cap each, then concat after the primary block
    order2 = np.argsort(second, kind="stable")          # groups tokens by 2nd cluster, cap each
    Vmap2_lo = order2.reshape(K, cap).astype(np.int64)
    Vmap2 = np.concatenate([Vmap, Vmap2_lo], axis=1)                  # (K, 2cap)
    Wperm2 = W[Vmap2.reshape(-1)].reshape(K, 2 * cap, D).astype(np.float16)
    np.savez(OUT, Cnorm=npz["Cnorm"], Wperm=Wperm2, Vmap=Vmap2)
    print(f"saved {OUT}  Wperm{Wperm2.shape} Vmap{Vmap2.shape}\n")

    # agreement: fp32 r=1 baseline vs r=2 (fp32 and int8-roundtrip weights)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=4000)
    print(f"{H.shape[0]} real WikiText-2 positions\n")

    Wperm2_q = int8_roundtrip(Wperm2.astype(np.float32))
    for P in (128, 256):
        a1 = (flash_top1(H, npz["Cnorm"], npz["Wperm"], Vmap, P) == dense).mean()
        a2 = (flash_top1(H, npz["Cnorm"], Wperm2, Vmap2, P) == dense).mean()
        a2q = (flash_top1(H, npz["Cnorm"], Wperm2_q, Vmap2, P) == dense).mean()
        print(f"P={P:<4d}  r=1 fp32 {a1:.2%}   r=2 fp32 {a2:.2%}   r=2 int8 {a2q:.2%}   "
              f"(r2 gain {a2-a1:+.2%}, int8 drop {a2q-a2:+.2%})")


if __name__ == "__main__":
    main()
