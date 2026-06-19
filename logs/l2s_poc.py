"""POC for RELATED.md: L2S (Learning to Screen) — does learned context-routing beat flat IVF?

L2S clusters the CONTEXT (hidden) vectors into G groups; each group gets a precomputed candidate
token set = the union of tokens actually observed for hidden states routed there. At inference,
route h to its group and score only that group's candidate set.

This differs from our data-aware-routing POC (which only moved the routing centroid): here the
candidate SET is learned from data, not a geometric cluster neighborhood.

Fit on the first positions, evaluate on held-out. Report recall@1 (== dense argmax) vs the mean
candidate-set size (the cost), against flat IVF at matched recall. A global frequent-token
shortlist is appended as the paper does, to backstop coverage.
"""
import numpy as np

npz = np.load("artifacts/qwen3_0_6b/clusters.npz")
Cnorm = npz["Cnorm"].astype(np.float32)
Vmap = npz["Vmap"]
D, K = Cnorm.shape
cap = Vmap.shape[1]
W = np.load("artifacts/qwen3_0_6b/head_W.npy")

from turbohead.eval.agreement import collect_hidden, wikitext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
H, dense = collect_hidden(model, tok, wikitext(240000), max_tokens=12000)  # exact argmax targets
H = H.astype(np.float32)
T = H.shape[0]
fit, ev = slice(0, 10000), slice(10000, T)
nf, ne = 10000, T - 10000
print(f"{T} positions | fit={nf} eval={ne} | V={W.shape[0]} K={K}\n")


def lloyd(X, G, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), G, replace=False)].copy()
    for _ in range(iters):
        a = ((X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)).argmin(1)
        for g in range(G):
            m = a == g
            if m.any():
                C[g] = X[m].mean(0)
    return C


# IVF baseline recall at matched candidate budgets (for reference)
sims = H[ev] @ Cnorm
print("flat IVF (FlashHead) reference:")
for P in (64, 128, 256):
    top = np.argpartition(-sims, P, axis=1)[:, :P]
    ok = sum(Vmap[top[i]].reshape(-1)[(W[Vmap[top[i]].reshape(-1)] @ H[ev][i]).argmax()] == dense[ev][i]
             for i in range(ne))
    print(f"  P={P:<4} cand={P * cap:<5} recall@1={ok / ne:.1%}")

# global frequent-token shortlist from fit (paper's backstop)
freq = np.bincount(dense[fit], minlength=W.shape[0])
print("\nL2S (context k-means + learned candidate sets, + top-F frequent backstop):")
print(f"  {'G':>4} {'F':>5} {'mean cand':>10} {'coverage':>9} {'recall@1':>9}")
for G in (64, 256):
    Cc = lloyd(H[fit], G)
    af = ((H[fit] ** 2).sum(1)[:, None] - 2 * H[fit] @ Cc.T + (Cc ** 2).sum(1)).argmin(1)
    ae = ((H[ev] ** 2).sum(1)[:, None] - 2 * H[ev] @ Cc.T + (Cc ** 2).sum(1)).argmin(1)
    learned = [np.unique(dense[fit][af == g]) for g in range(G)]   # candidate set per group
    for F in (0, 256, 1024):
        backstop = np.argsort(-freq)[:F]
        cand = [np.union1d(learned[g], backstop) for g in range(G)]
        sizes, cov, ok = [], 0, 0
        for i in range(ne):
            S = cand[ae[i]]
            sizes.append(len(S))
            if dense[ev][i] in S:
                cov += 1
                pred = S[(W[S] @ H[ev][i]).argmax()]
                ok += pred == dense[ev][i]
        print(f"  {G:>4} {F:>5} {np.mean(sizes):>10.0f} {cov / ne:>8.1%} {ok / ne:>8.1%}")
