"""POC for IDEAS.md #1: hierarchical (coarse-to-fine) stage-1 routing.

Flat stage 1 scores all K centroids (C.h), cost ~K*D. Hierarchical: cluster the K centroids
into M super-centroids, score those (M*D), descend into the top-m super-clusters, score only
their leaf centroids (~m*(K/M)*D). Big FLOP/byte cut IF recall holds: the true cluster's
super-cluster must be in the top-m.

We measure both: stage-1 cost reduction, and top-1 agreement vs the FLAT baseline at P=256
(agreement = the true cluster is reachable and ranks within top-P among candidate leaves).
"""
import sys
import numpy as np
from turbohead.eval.agreement import collect_hidden, wikitext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
npz = np.load("artifacts/qwen3_0_6b/clusters.npz")
Cnorm, Vmap = npz["Cnorm"].astype(np.float32), npz["Vmap"]   # Cnorm:(D,K), Vmap:(K,cap)
D, K = Cnorm.shape
cap = Vmap.shape[1]
C = Cnorm.T                                                   # (K, D) one row per cluster centroid

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=4000)
T = H.shape[0]
tok2clu = np.empty(K * cap, np.int64)
for k in range(K):
    tok2clu[Vmap[k]] = k
true_clu = tok2clu[dense]                                     # (T,) true cluster per position

def lloyd(X, M, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    SC = X[rng.choice(len(X), M, replace=False)].copy()
    for _ in range(iters):
        a = (X @ SC.T).argmax(1)                              # cosine-ish: C rows ~unit norm
        for m in range(M):
            sel = a == m
            if sel.any():
                v = X[sel].mean(0); SC[m] = v / (np.linalg.norm(v) + 1e-9)
    return SC

P = 256
flat_sims = H @ Cnorm                                         # (T, K)
flat_rank = (flat_sims > flat_sims[np.arange(T), true_clu][:, None]).sum(1) + 1
flat_agree = (flat_rank <= P).mean()
print(f"{T} positions | K={K} cap={cap} | FLAT baseline agree@P={P}: {flat_agree:.1%}\n")
print(f"r = leaf assigned to its top-r super-clusters (soft assignment)\n")
print(f"{'M':>4} {'r':>3} {'m':>3} {'stage1 cost vs flat':>20} {'reachable':>10} {'agree@256':>10} {'vs flat':>8}")

for M in (100, 256):
    SC = lloyd(C, M)                                          # (M, D)
    leaf_super_sims = C @ SC.T                                # (K, M) each leaf vs each super
    super_sims = H @ SC.T                                     # (T, M)
    super_order = np.argsort(-super_sims, axis=1)
    for r in (1, 2, 3):
        leaf2supers = np.argsort(-leaf_super_sims, axis=1)[:, :r]   # (K, r) supers per leaf
        # membership[leaf, super] -> is this super one of the leaf's r homes
        member = np.zeros((K, M), bool)
        member[np.arange(K)[:, None], leaf2supers] = True
        true_supers = leaf2supers[true_clu]                  # (T, r)
        for m in (4, 8, 16):
            top_supers = super_order[:, :m]                  # (T, m)
            sel_super = np.zeros((T, M), bool)
            sel_super[np.arange(T)[:, None], top_supers] = True
            reachable = (sel_super[np.arange(T)[:, None], true_supers]).any(1)
            cand_count = (sel_super @ member.T > 0)          # (T, K) leaf is candidate
            cost = (M + cand_count.sum(1).mean()) / K
            agree = 0
            for i in range(T):
                if not reachable[i]:
                    continue
                cand = cand_count[i]
                ts = flat_sims[i, true_clu[i]]
                rank = (flat_sims[i, cand] > ts).sum() + 1
                agree += rank <= P
            agree /= T
            print(f"{M:>4} {r:>3} {m:>3} {1/cost:>17.1f}x {reachable.mean():>10.1%} "
                  f"{agree:>10.1%} {agree/flat_agree:>7.2f}x")
