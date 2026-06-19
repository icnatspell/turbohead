"""POC for IDEAS.md #3: MIPS-aware cluster ranking.

Stage 1 ranks clusters by COSINE (normalized centroid . h). But the real target is the cluster
holding argmax(e . h), a raw inner product. Normalizing the centroid discards embedding-norm
info, and in maximum-inner-product search that norm is exactly what flags a cluster as possibly
holding a high-scoring token.

We keep the cluster GROUPS fixed and only change the stage-1 routing score, then recompute
required-P (rank of the true cluster) on the same hidden states. Lower required-P = win.

Variants tested:
  baseline   : (mean_e / ||mean_e||) . h          [current FlashHead, cosine]
  raw_mean   : mean_e . h                          [unnormalized -> norm-weighted]
  mean+rad   : mean_e . h + r_k * ||h||            [MIPS upper bound, r_k = max member dist to mean]
  mean+maxn  : mean_e . h + (max_i||e_i||) * ||h|| [cruder cap on the cluster's best possible IP]
"""
import numpy as np

npz = np.load("artifacts/qwen3_0_6b/clusters.npz")
Cnorm = npz["Cnorm"].astype(np.float32)        # (D, K) normalized mean embeddings (cosine routing)
Wperm = npz["Wperm"].astype(np.float32)        # (K, cap, D) member embeddings per cluster
Vmap = npz["Vmap"]                             # (K, cap)
D, K = Cnorm.shape
cap = Wperm.shape[1]

# per-cluster geometry
mu = Wperm.mean(1)                             # (K, D) raw mean embedding
dist = np.linalg.norm(Wperm - mu[:, None, :], axis=2)   # (K, cap) member dist to mean
rad = dist.max(1)                              # (K,) cluster radius
maxn = np.linalg.norm(Wperm, axis=2).max(1)    # (K,) max member embedding norm

from turbohead.eval.agreement import collect_hidden, wikitext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=4000)   # H:(T,D), dense:(T,)
T = H.shape[0]
hnorm = np.linalg.norm(H, axis=1)              # (T,)

tok2clu = np.empty(K * cap, np.int64)
for k in range(K):
    tok2clu[Vmap[k]] = k
true_clu = tok2clu[dense]                       # (T,) true cluster per position


def required_p(sims):
    """sims:(T,K) routing scores. Rank of true cluster (1 = best). Higher score = better."""
    ts = sims[np.arange(T), true_clu]
    return (sims > ts[:, None]).sum(1) + 1


def report(name, sims):
    rank = required_p(sims)
    pct = {p: int(np.percentile(rank, p)) for p in (50, 90, 95, 99)}
    accs = {P: f"{(rank <= P).mean():.1%}" for P in (64, 128, 256)}
    print(f"  {name:11s} p50={pct[50]:<3} p90={pct[90]:<4} p95={pct[95]:<5} p99={pct[99]:<6}"
          f" mean={rank.mean():7.1f} | @64={accs[64]} @128={accs[128]} @256={accs[256]}")


print(f"{T} positions | K={K} cap={cap}\n")
report("baseline", H @ Cnorm)
report("raw_mean", H @ mu.T)
ip = H @ mu.T                                   # (T, K)
report("mean+rad", ip + hnorm[:, None] * rad[None, :])
report("mean+maxn", ip + hnorm[:, None] * maxn[None, :])

# the bound terms blow up at full strength (they just sort by radius). sweep a small coeff.
print("\nradius-term coefficient sweep on raw inner product (ip + a*||h||*rad):")
for a in (0.0, 0.01, 0.03, 0.1, 0.3):
    report(f"a={a}", ip + a * hnorm[:, None] * rad[None, :])
