r"""POC: FACTORIZED (PQ) ROUTER — can a cheaper stage-1 preserve cluster ranking, to afford a bigger P?

WHY (junior-engineer version)
-----------------------------
Stage 1 is the DOMINANT head cost: scoring h against all K centroids is a [K,D] gemv (here 9496 x
1024). Agreement is recall-limited, and the cheapest way to lift recall is to probe more clusters
(raise P) -- but P is gated by how much stage-1 budget we have. So: if we can compute the SAME cluster
ranking with far fewer FLOPs, we can raise P at constant cost, and from the required-P tail
(p99 ~1165) a bigger P captures most of the remaining misses. That's the lever.

Classical tool: Product Quantization with Asymmetric Distance Computation (Jegou et al., PAMI 2011;
the inverted multi-index, Babenko & Lempitsky CVPR 2012, is the routing form). Split each centroid
into m sub-vectors; quantize each subspace into a tiny codebook of Ksub codewords. To route a query:
- build m lookup tables, LUT[s][j] = h_sub_s . codeword_sj   (cost m*Ksub*Dsub)
- approximate each centroid's score as a sum of m table lookups   (cost K*m adds)
Total ~ Ksub*D + K*m  vs the exact K*D. With Ksub=256, m=8: ~28x fewer multiply-adds.

THE RISK (why it might not work)
--------------------------------
PQ approximates h.c_k. If the approximation reorders clusters near the top-P boundary, the true
cluster can fall out of top-P and recall DROPS. hierarchical_stage1 already showed coarse routing
"drops the hard tail". So this is a measurement of a tradeoff, not a free win: we want PQ recall@P to
stay close to exact while the FLOP ratio is large enough that "raise P" more than pays it back.

NOTE: the shipped stage-1 already runs as int4 MatMulNBits (a fast quantized gemv). PQ-ADC is a
DIFFERENT mechanism (table lookups, cache-bound, not a gemv) -- whether it actually beats int4 in
ORT is an op-level question this numpy POC can't answer; here we measure the FLOP ratio and the
recall cost. If recall holds, the ORT speed test is the graduation step.

Read-only research: imports core helpers, touches nothing in turbohead/.
Run: uv run python experimental/factorized_router/factorized_router_poc.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import collect_hidden, wikitext

MODEL = "Qwen/Qwen3-0.6B"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
MAX_TOKENS = 4000
KSUB = 256                       # codewords per subspace (fits a uint8 code)
MS = (4, 8, 16)                  # subspace counts to sweep
PS = (128, 256, 512, 1024)
SEED = 0


def pq_train(C, m, ksub, iters=10, seed=0):
    """Product-quantize the (K,D) centroid matrix: per subspace, k-means -> (ksub,Dsub) codebook and
    a (K,) code. Returns codebooks list and codes (K,m)."""
    K, D = C.shape
    dsub = D // m
    rng = np.random.default_rng(seed)
    books, codes = [], np.empty((K, m), np.int64)
    for s in range(m):
        X = C[:, s * dsub:(s + 1) * dsub]            # (K,Dsub)
        cb = X[rng.choice(K, ksub, replace=False)].copy()
        for _ in range(iters):
            a = (X @ cb.T - 0.5 * (cb * cb).sum(1)).argmax(1)   # nearest codeword (argmin L2)
            for j in range(ksub):
                sel = a == j
                if sel.any():
                    cb[j] = X[sel].mean(0)
        books.append(cb)
        codes[:, s] = (X @ cb.T - 0.5 * (cb * cb).sum(1)).argmax(1)
    return books, codes


def pq_scores(H, books, codes, m):
    """Approximate H @ C.T via ADC: per subspace LUT = H_sub @ codebook.T, sum the looked-up codes."""
    D = H.shape[1]
    dsub = D // m
    out = np.zeros((H.shape[0], codes.shape[0]), np.float32)
    for s in range(m):
        lut = H[:, s * dsub:(s + 1) * dsub] @ books[s].T     # (T,ksub)
        out += lut[:, codes[:, s]]                           # gather code column -> (T,K)
    return out


def required_p(sims, true_clu):
    ts = sims[np.arange(sims.shape[0]), true_clu]
    return (sims > ts[:, None]).sum(1) + 1


def agree_line(name, rank, flops_ratio):
    acc = {P: f"{(rank <= P).mean():.2%}" for P in PS}
    print(f"  {name:16s} @128={acc[128]} @256={acc[256]} @512={acc[512]} @1024={acc[1024]} "
          f"| stage-1 FLOPs vs exact: {flops_ratio}")


def main():
    npz = np.load(NPZ)
    Cnorm = npz["Cnorm"].astype(np.float32)          # (D,K)
    Vmap = npz["Vmap"]
    D, K = Cnorm.shape
    cap = Vmap.shape[1]
    C = Cnorm.T.copy()                               # (K,D) unit centroids (what cosine routing uses)
    tok2clu = np.empty(K * cap, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=MAX_TOKENS)
    true_clu = tok2clu[dense]
    print(f"{H.shape[0]} positions | K={K} cap={cap} D={D} | PQ ksub={KSUB}\n")
    print("recall of the TRUE cluster; PQ must stay near exact while FLOPs drop enough to raise P.\n")

    agree_line("exact (cosine)", required_p(H @ Cnorm, true_clu), "1.0x (K*D)")
    for m in MS:
        books, codes = pq_train(C, m, KSUB)
        rank = required_p(pq_scores(H, books, codes, m), true_clu)
        ratio = (K * D) / (KSUB * D + K * m)         # exact / approx multiply-adds
        agree_line(f"PQ m={m}", rank, f"~{ratio:.0f}x fewer")


if __name__ == "__main__":
    main()
