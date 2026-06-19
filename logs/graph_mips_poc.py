"""POC for RELATED.md: is a graph index (HNSW MIPS) more candidate-efficient than our flat IVF?

Both methods predict the head's top-1 by scoring some token rows; the cost that matters is the
number of inner products computed per query. We compare on ONE axis: top-1 recall (== dense
argmax) versus distance-computations-per-query.

  flat IVF (FlashHead): cost = K (stage-1 centroid scan) + P*cap (stage-2 shortlist). Fixed K
                        floor regardless of P.
  HNSW MIPS           : navigates a graph over the V token embeddings directly; faiss reports the
                        exact distance-computation count (hnsw_stats.ndis). No fixed floor.

If HNSW reaches ~the same recall@1 at far fewer inner products than IVF's K + P*cap, the graph
direction has promise (better recall per candidate -> lower P at fixed top-1).

Caveat: HNSW with inner product is not a true metric (no triangle inequality), so its recall is a
slight under-estimate of what an MIPS-tuned graph would give. Indicative, not final.
"""
import sys
import time
import numpy as np
import faiss

npz = np.load("artifacts/qwen3_0_6b/clusters.npz")
Cnorm = npz["Cnorm"].astype(np.float32)        # (D, K)
Vmap = npz["Vmap"]                             # (K, cap)
D, K = Cnorm.shape
cap = Vmap.shape[1]
W = np.ascontiguousarray(np.load("artifacts/qwen3_0_6b/head_W.npy")).astype(np.float32)  # (V, D)
V = W.shape[0]

from turbohead.eval.agreement import collect_hidden, wikitext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
H, dense = collect_hidden(model, tok, wikitext(80000), max_tokens=2000)   # dense = exact argmax
H = np.ascontiguousarray(H.astype(np.float32))
T = H.shape[0]
print(f"{T} queries | V={V} D={D} K={K} cap={cap}\n")

# --- flat IVF (our current FlashHead): recall@1 vs cost at several P ---
sims = H @ Cnorm                               # (T, K)
print("flat IVF (FlashHead):")
print(f"  {'P':>4} {'candidates':>10} {'cost (dots/q)':>14} {'recall@1':>9}")
for P in (64, 128, 256, 512):
    top = np.argpartition(-sims, P, axis=1)[:, :P]      # (T, P) clusters
    correct = 0
    for i in range(T):
        ids = Vmap[top[i]].reshape(-1)                  # (P*cap,)
        pred = ids[(W[ids] @ H[i]).argmax()]
        correct += pred == dense[i]
    cost = K + P * cap
    print(f"  {P:>4} {P*cap:>10} {cost:>14} {correct / T:>8.1%}")

# --- HNSW MIPS over the V token embeddings ---
print("\nbuilding HNSW index (M=32)...")
t0 = time.time()
index = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 80
index.add(W)
print(f"  built in {time.time() - t0:.1f}s")

print("\nHNSW MIPS (recall@1 = top-1 graph result == exact argmax):")
print(f"  {'efSearch':>9} {'cost (dots/q)':>14} {'recall@1':>9}")
for ef in (32, 64, 128, 256, 512, 1024):
    index.hnsw.efSearch = ef
    faiss.cvar.hnsw_stats.reset()
    _, I = index.search(H, 1)
    ndis = faiss.cvar.hnsw_stats.ndis / T
    recall = (I[:, 0] == dense).mean()
    print(f"  {ef:>9} {ndis:>14.0f} {recall:>8.1%}")
