"""POC: a DISCRIMINATIVELY-trained linear routing metric — the one unfalsified routing transform.

Background. Agreement == stage-1 recall: FlashHead returns the dense argmax exactly when that token's
cluster lands in the top-P clusters by routing score (stage 2 is exact). "required-P" of a position =
the rank of its true cluster in the routing order; agree@P = fraction with required-P <= P. So a
better routing *score* lifts agreement at zero extra runtime cost — IF it stays a single K x D gemv.

What's already falsified here (don't repeat):
  * recall_lift: mean-centering, diagonal whitening, learned-per-cluster-h prototype — all lose.
  * whitened_routing: the GENERATIVE Mahalanobis metric A=(Sigma+lambda*I)^-1 — loses monotonically
    (high-variance hidden dims are signal, whitening deletes them).
  * factorized_router: PQ-approximated routing — loses (recall is razor-sensitive at the top-P cut).
  * A pure ORTHOGONAL rotation R is a provable NO-OP for single-codebook cosine routing:
    cos(Rh, R mu_k) = cos(h, mu_k) exactly (R preserves every inner product). OPQ-style rotation only
    bites paired with a product quantizer. So "rotate before clustering" cannot help on its own.

The empty cell. Every linear-metric test above was UNSUPERVISED (a fixed covariance/whitening choice).
None trained the metric on the recall objective itself. ScaNN's actual contribution (Guo et al.,
ICML 2020) is a *score-aware* loss; anisotropic_clustering applied it to the PARTITION (the --eta
knob, graduated). This applies the discriminative version to the QUERY METRIC: learn a general linear
map L (D x D) by gradient descent so the true cluster's cosine score ranks in the top-P, then FOLD L
into the centroids (c'_k = normalize(L mu_k)) so inference is the identical h . c' gemv — free.

This is NOT an approximation of the routing inner product (the meta-finding that killed PQ/whitening
warns against that) — it computes an EXACT inner product against relearned centroids. The honest prior
from whitened_routing is "this probably loses", but a recall-trained L is its steelman and is cheap.

Read-only research; touches nothing shipped. Run: uv run python experimental/learned_metric/learned_metric_poc.py
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import wikitext

MODEL = "Qwen/Qwen3-0.6B"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
CACHE = "/tmp/learned_metric_HD.npz"   # H + dense argmax cache (the only slow part: 6 model forwards)
CHUNK, N_CHUNKS, SEED = 2000, 6, 0
STEPS, BATCH, NEG = 500, 1024, 2048    # sampled-softmax: per step score true clusters + NEG randoms


def collect():
    """Per position: h (lm_head input) and the dense argmax token. Cached — the model forwards are
    the only expensive part; training below is pure torch on the cached arrays."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["H"], z["dense"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    big = wikitext(CHUNK * N_CHUNKS * 12)
    step = len(big) // N_CHUNKS
    Hs, ds = [], []
    for i in range(N_CHUNKS):
        h_in = {}
        hook = model.lm_head.register_forward_hook(lambda m, i, o: h_in.update(h=i[0]))
        ids = tok(big[i * step:(i + 1) * step], return_tensors="pt").input_ids[:, :CHUNK]
        with torch.no_grad():
            logits = model(ids).logits[0].float().numpy()
        hook.remove()
        Hs.append(h_in["h"][0].float().numpy())
        ds.append(logits.argmax(-1))
    H, dense = np.concatenate(Hs), np.concatenate(ds)
    np.savez(CACHE, H=H, dense=dense)
    return H, dense


def required_p(sims, true_clu):
    ts = sims[np.arange(sims.shape[0]), true_clu]
    return (sims > ts[:, None]).sum(1) + 1


def report(name, rank, note):
    pct = {p: int(np.percentile(rank, p)) for p in (50, 90, 99)}
    acc = {P: f"{(rank <= P).mean():.2%}" for P in (128, 256, 512)}
    print(f"  {name:18s} p50={pct[50]:<3} p90={pct[90]:<4} p99={pct[99]:<6} "
          f"| @128={acc[128]} @256={acc[256]} @512={acc[512]}   [{note}]")


def train_metric(H, true_clu, mu, tr, rank, lowrank=0):
    """Learn L (D x D) so cosine(h, L mu_{true}) ranks the true cluster high. Sampled-softmax CE over
    the batch's true clusters + NEG random clusters. lowrank>0: L = I + U V^T (fewer params, less
    overfit). Returns the folded routing matrix Cnew (D,K) = normalize(L mu)^T, drop-in for Cnorm."""
    g = torch.Generator().manual_seed(SEED)
    Ht = torch.from_numpy(H[tr]).float()
    yt = torch.from_numpy(true_clu[tr]).long()
    MU = torch.from_numpy(mu).float()                      # (K,D) raw mean embedding
    D, K = mu.shape[1], mu.shape[0]
    if lowrank:
        U = torch.zeros(D, lowrank, requires_grad=True)
        V = torch.zeros(D, lowrank, requires_grad=True)
        torch.nn.init.normal_(U, std=1e-3)
        torch.nn.init.normal_(V, std=1e-3)
        params, Lfn = [U, V], lambda: torch.eye(D) + U @ V.T
    else:
        Lp = torch.eye(D, requires_grad=True)
        params, Lfn = [Lp], lambda: Lp
    scale = torch.tensor(10.0, requires_grad=True)         # cosine logits live in [-1,1]; learn temp
    opt = torch.optim.Adam(params + [scale], lr=2e-3)
    Hn = Ht / Ht.norm(dim=1, keepdim=True)                 # h-normalization is a per-row const (rank-free)
    n = len(tr)
    for s in range(STEPS):
        bi = torch.randint(0, n, (BATCH,), generator=g)
        ytb = yt[bi]
        neg = torch.randint(0, K, (NEG,), generator=g)
        cand = torch.cat([ytb, neg]).unique()             # candidate clusters this step
        pos = torch.searchsorted(cand, ytb)               # label = position of true cluster in cand
        L = Lfn()
        Lmu = MU[cand] @ L.T                               # (|cand|,D) = L mu_k
        Lmu = Lmu / (Lmu.norm(dim=1, keepdim=True) + 1e-9)
        logits = scale * (Hn[bi] @ Lmu.T)                  # (BATCH,|cand|) cosine scores
        loss = torch.nn.functional.cross_entropy(logits, pos)
        loss = loss + 1e-3 * ((L - torch.eye(D)) ** 2).sum()   # shrink toward identity (cosine)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 100 == 0 or s == STEPS - 1:
            print(f"    step {s:3d}  loss {loss.item():.3f}")
    with torch.no_grad():
        Cnew = (MU @ Lfn().T)
        Cnew = Cnew / (Cnew.norm(dim=1, keepdim=True) + 1e-9)
    return Cnew.numpy().T                                  # (D,K), like Cnorm


def main():
    rng = np.random.default_rng(SEED)
    npz = np.load(NPZ)
    Cnorm = npz["Cnorm"].astype(np.float32)               # (D,K)
    Vmap = npz["Vmap"]
    mu = npz["Wperm"].astype(np.float32).mean(1)          # (K,D) raw mean embedding
    D, K = Cnorm.shape
    cap = Vmap.shape[1]
    tok2clu = np.empty(K * cap, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)

    H, dense = collect()
    true_clu = tok2clu[dense]
    T = H.shape[0]
    perm = rng.permutation(T)
    tr, te = perm[: T // 2], perm[T // 2:]
    Hte, cte = H[te], true_clu[te]
    print(f"{T} positions ({len(tr)} fit / {len(te)} eval) | K={K} cap={cap} D={D}\n")

    rank_cos = required_p(Hte @ Cnorm, cte)
    print("baseline:")
    report("cosine (current)", rank_cos, "shipped")

    print("\nlearned full-rank metric L (D x D):")
    Cfull = train_metric(H, true_clu, mu, tr, rank_cos, lowrank=0)
    report("learned-L full", required_p(Hte @ Cfull, cte), "free (folded into centroids)")

    print("\nlearned low-rank metric L = I + U V^T (r=32):")
    Clow = train_metric(H, true_clu, mu, tr, rank_cos, lowrank=32)
    report("learned-L r=32", required_p(Hte @ Clow, cte), "free (folded into centroids)")


if __name__ == "__main__":
    main()
