"""Throwaway analysis (thesis dir #1 + #4 scoping). Oracle headroom for adaptive probing:
per-token minimum P = rank of the dense-argmax token's cluster in the centroid-logit order.
Also a quick #4 probe: is the high-rank tail correlated with token rarity?"""
import sys
import numpy as np
from collections import Counter
from turbohead.eval.agreement import collect_hidden, wikitext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
npz = np.load("artifacts/qwen3_0_6b/clusters.npz")
Cnorm, Vmap = npz["Cnorm"].astype(np.float32), npz["Vmap"]   # Cnorm:(D,K), Vmap:(K,cap)
K, cap = Vmap.shape

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=4000)   # H:(T,D), dense:(T,)
T = H.shape[0]
print(f"{T} positions | K={K} clusters cap={cap}")

# token -> cluster reverse map
tok2clu = np.empty(K * cap, np.int64)
for k in range(K):
    tok2clu[Vmap[k]] = k

sims = H @ Cnorm                       # (T, K) centroid logits
# rank of the true cluster = how many clusters score strictly higher than it (0-based -> +1)
true_clu = tok2clu[dense]             # (T,)
true_sim = sims[np.arange(T), true_clu]
rank = (sims > true_sim[:, None]).sum(1) + 1   # min P that includes the right cluster

pct = [50, 90, 95, 99, 99.9]
print("rank percentiles:", {p: int(np.percentile(rank, p)) for p in pct}, "max", int(rank.max()))
print("mean rank (=oracle avg P at 100% acc):", f"{rank.mean():.1f}")
print()
print(" fixed-P accuracy   vs   oracle-adaptive avg-P for same accuracy")
for P in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    acc = (rank <= P).mean()
    # adaptive: spend exactly rank[i] probes on tokens we'd get (rank<=P), cap others at P
    avg_p_adaptive = np.minimum(rank, P).mean()
    print(f"  P={P:<4d} acc={acc:6.2%}   fixed cost={P:<4d}  adaptive avg-P={avg_p_adaptive:6.1f}")

# #4 scoping: is the hard tail (high rank) driven by rare tokens?
freq = Counter(dense.tolist())
tok_freq = np.array([freq[t] for t in dense])
easy = rank <= 8
hard = rank > 64
print()
print(f"#4 probe: median corpus-freq of argmax token | easy(rank<=8): {np.median(tok_freq[easy]):.0f}"
      f" | hard(rank>64): {np.median(tok_freq[hard]):.0f}  (n_hard={hard.sum()})")
