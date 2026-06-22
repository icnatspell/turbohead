"""POC for IDEAS.md #4: coverage-corrected probabilities.

Today FlashHead scores only the P*cap probed candidates; everything else gets logit -1e9, so a
true token in an UNPROBED cluster gets ~0 probability. That caps full-distribution PPL and
distorts sampling. This is the one measured quality deficiency in the method.

Fix: each stage-1 cluster has a mean-embedding centroid mu_k, so mu_k . h is the cluster's MEAN
logit. Use it to (a) add the unprobed clusters' mass to the softmax denominator Z, and (b) give
an unprobed true token a non-zero probability instead of ~0. No extra weight reads at deploy
(store the per-cluster norm; here we just compute mu @ h, same cost as stage 1).

We compare, on real hidden states, NLL/PPL against the exact full-V softmax (gold):
  gold       : full softmax over all V exact logits (the dense head's own PPL)
  trunc      : current FlashHead (prob 0 outside the probed set)
  corr_full  : DEPLOYABLE fix — corrected Z, unprobed true token gets its cluster mean logit
  corr_Zonly : ceiling — corrected Z, but cheat the unprobed true token's EXACT logit
               (isolates denominator error from numerator error)
"""
import numpy as np


def logsumexp(a, axis=None):
    m = a.max(axis=axis, keepdims=True)
    out = np.log(np.exp(a - m).sum(axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)

P = 256
T = 1000
CHUNK = 100

npz = np.load("artifacts/qwen3_0_6b/clusters.npz")
Cnorm = npz["Cnorm"].astype(np.float32)        # (D, K) cosine routing centroids
Wperm = npz["Wperm"].astype(np.float32)        # (K, cap, D)
Vmap = npz["Vmap"]                             # (K, cap)
D, K = Cnorm.shape
cap = Vmap.shape[1]
mu = Wperm.mean(1).astype(np.float32)          # (K, D) raw mean embedding -> mean-logit centroid
meannorm = np.linalg.norm(Wperm, axis=2).mean(1).astype(np.float64)   # (K,) typical member norm

W = np.load("artifacts/qwen3_0_6b/head_W.npy", mmap_mode="r")   # (V, D) fp32

from turbohead.eval.agreement import wikitext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)

# capture lm_head input + the GENUINE corpus next token (not the model's argmax) so PPL actually
# exercises the coverage hole: real next tokens are often rare and land in unprobed clusters.
h_in = {}
hook = model.lm_head.register_forward_hook(lambda m, i, o: h_in.update(h=i[0]))
ids = tok(wikitext(60000), return_tensors="pt").input_ids[:, : T + 1]
with torch.no_grad():
    model(ids)
hook.remove()
H = h_in["h"][0].float().numpy()[:-1]           # (T, D) hidden at position t
y = ids[0, 1:].numpy().astype(np.int64)         # (T,) genuine next token (corpus)
T = H.shape[0]

tok2clu = np.empty(K * cap, np.int64)
for k in range(K):
    tok2clu[Vmap[k]] = k
true_clu = tok2clu[y]

nll = {m: np.empty(T) for m in ("gold", "trunc", "corr_full", "corr_norm", "corr_cal", "corr_Zonly")}
probed_flag = np.empty(T, bool)
W = np.asarray(W)                              # materialize once (622 MB)

for s in range(0, T, CHUNK):
    e = min(s + CHUNK, T)
    Hc = H[s:e]                                # (c, D)
    c = e - s
    L = (W @ Hc.T).astype(np.float64)          # (V, c) exact logits
    sims = (Cnorm.T @ Hc.T).astype(np.float64)  # (K, c) cosine routing
    m = (mu @ Hc.T).astype(np.float64)         # (K, c) per-cluster mean logits
    top = np.argpartition(-sims, P, axis=0)[:P]    # (P, c) probed clusters

    logZ_gold = logsumexp(L, axis=0)           # (c,)
    expm = np.exp(m)                           # (K, c)
    tail_all = cap * expm.sum(0)               # (c,) cap*sum_k exp(mean_k)

    for j in range(c):
        i = s + j
        yi, cl = y[i], true_clu[i]
        nll["gold"][i] = -L[yi, j] + logZ_gold[j]

        clusters = top[:, j]
        ids = Vmap[clusters].reshape(-1)               # (P*cap,) probed token ids
        Zp = np.exp(L[ids, j]).sum()                   # probed partition (exact)
        probed = cl in set(clusters.tolist())
        probed_flag[i] = probed

        # truncated (current): prob 0 outside probed set
        if probed:
            nll["trunc"][i] = -L[yi, j] + np.log(Zp)
        else:
            nll["trunc"][i] = -np.log(1e-30)           # floored ~0 prob

        # corrected Z: probed clusters exact (in Zp), unprobed clusters via mean-logit proxy
        tail = tail_all[j] - cap * expm[clusters, j].sum()   # exclude probed clusters
        Zc = Zp + tail
        if probed:
            nll["corr_full"][i] = nll["corr_norm"][i] = nll["corr_Zonly"][i] = -L[yi, j] + np.log(Zc)
        else:
            nll["corr_full"][i] = -m[cl, j] + np.log(Zc)             # deployable: cluster mean logit
            nll["corr_norm"][i] = -sims[cl, j] * meannorm[cl] + np.log(Zc)  # cosine * typical norm
            nll["corr_Zonly"][i] = -L[yi, j] + np.log(Zc)           # ceiling: cheat exact logit

# the numerator gap: unprobed true tokens sit ABOVE their cluster mean logit (exp is convex,
# and the corpus wants them). Test whether one global offset delta, fit offline, closes it.
gap = np.array([
    nll["corr_full"][i] - nll["corr_Zonly"][i] for i in range(T) if not probed_flag[i]
])
delta = gap.mean()                              # = mean(L[y] - m_cl) over unprobed positions
nll["corr_cal"] = nll["corr_full"].copy()
nll["corr_cal"][~probed_flag] -= delta          # m_cl + delta as the numerator
print(f"fitted numerator offset delta = {delta:.2f} nats\n")

print(f"{T} positions | P={P} cap={cap} K={K} | coverage (true token probed): {probed_flag.mean():.1%}\n")
print(f"{'method':12s} {'PPL':>10} {'mean NLL':>10}   {'PPL (probed-only)':>18} {'PPL (unprobed-only)':>20}")
for name in ("gold", "trunc", "corr_full", "corr_norm", "corr_cal", "corr_Zonly"):
    v = nll[name]
    ppl = np.exp(v.mean())
    pp = np.exp(v[probed_flag].mean())
    pu = np.exp(v[~probed_flag].mean()) if (~probed_flag).any() else float("nan")
    print(f"{name:12s} {ppl:>10.2f} {v.mean():>10.3f}   {pp:>18.2f} {pu:>20.2f}")
