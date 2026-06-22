"""POC: targeted SECOND HOME vs the shipped ALWAYS-SCORE lever — same miss tail, two mechanisms.

WHY (read this first, junior-engineer version)
----------------------------------------------
Top-1 agreement == stage-1 recall: FlashHead returns the dense argmax token only when that token's
cluster lands in the top-P probed clusters. The misses are a heavy tail, and about half of them are
the SAME frequent tokens every time (function words, punctuation) that sit far from their cluster's
centroid, so routing never ranks them in. Two ways to rescue exactly those tokens:

  ALWAYS-SCORE (shipped, recall_lift lever 4). Find the most-missed tokens offline
  (`turbohead-calibrate-misses`), then score their weight rows on EVERY step regardless of routing
  (they ride the graph's existing Wspec/EOS path). Cost: a flat N extra rows scored every step.
  Rescue: unconditional — if the dense token is in the list, it is always caught.

  TARGETED SECOND HOME (this POC). Give those same tokens a SECOND home cluster (their next-best
  cluster by embedding direction), so they become reachable through a second route. Cost: a token's
  extra row is gathered ONLY on steps where its second cluster is in the top-P — amortized, usually
  far below N. Rescue: conditional — caught only when that second cluster happens to be probed.

So always-score pays a flat cost for unconditional rescue; second home pays a tiny amortized cost for
conditional rescue. This POC measures both on the SAME held-out split at the deploy P=256: rescue
fraction, resulting agreement, and the extra-rows-per-step cost each one charges.

It is the honest framing for the "targeted second home" idea: it does not compete with a blanket r=2
(which doubles every cluster), it competes with the always-score lever already in the graph.

Read-only research: imports core helpers but touches nothing in turbohead/.
Run: uv run python experimental/targeted_second_home/targeted_second_home_poc.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import wikitext

MODEL = "Qwen/Qwen3-0.6B"
HEAD = "artifacts/qwen3_0_6b/head_W.npy"     # (V,D) raw lm_head rows = token embeddings
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
CHUNK = 2000           # tokens per forward pass
N_CHUNKS = 6           # ~12k positions, split 50/50 fit / eval
P = 256                # deploy probe budget
SEED = 0


def collect(model, tok, texts):
    """Per position: h (input to lm_head) and the dense argmax token. One forward per text slice."""
    Hs, ds = [], []
    for text in texts:
        h_in = {}
        hook = model.lm_head.register_forward_hook(lambda m, i, o: h_in.update(h=i[0]))
        ids = tok(text, return_tensors="pt").input_ids[:, :CHUNK]
        with torch.no_grad():
            logits = model(ids).logits[0].float().numpy()   # (T,V) transient
        hook.remove()
        Hs.append(h_in["h"][0].float().numpy())
        ds.append(logits.argmax(-1))
    return np.concatenate(Hs), np.concatenate(ds)


def required_p(sims, true_clu):
    ts = sims[np.arange(sims.shape[0]), true_clu]
    return (sims > ts[:, None]).sum(1) + 1


def second_homes(tokens, W, Cnorm, tok2clu):
    """For each token, its best home OTHER than its primary cluster: the next-nearest centroid by
    embedding direction. That is the cluster most likely to rank high when this token is the answer
    (the hidden state arrives roughly along the token's own embedding)."""
    St = W[tokens] @ Cnorm                       # (n,K)  token-embedding vs every centroid
    order = np.argsort(-St, axis=1)
    prim = tok2clu[tokens]
    top1 = order[:, 0]
    return np.where(top1 != prim, top1, order[:, 1])   # skip primary if it is the nearest


def main():
    rng = np.random.default_rng(SEED)
    npz = np.load(NPZ)
    Cnorm = npz["Cnorm"].astype(np.float32)      # (D,K)
    Vmap = npz["Vmap"]
    D, K = Cnorm.shape
    cap = Vmap.shape[1]
    W = np.load(HEAD).astype(np.float32)         # (V,D)
    V = W.shape[0]
    tok2clu = np.empty(V, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    big = wikitext(CHUNK * N_CHUNKS * 12)
    step = len(big) // N_CHUNKS
    H, dense = collect(model, tok, [big[i * step:(i + 1) * step] for i in range(N_CHUNKS)])
    true_clu = tok2clu[dense]
    T = H.shape[0]

    perm = rng.permutation(T)
    tr, te = perm[: T // 2], perm[T // 2:]
    sims_te = H[te] @ Cnorm                       # cosine routing on eval
    base = required_p(sims_te, true_clu[te]) <= P
    thresh = np.partition(sims_te, K - P, axis=1)[:, K - P]   # P-th largest sim per eval row
    print(f"{T} positions ({len(tr)} fit / {len(te)} eval) | K={K} cap={cap} P={P}")
    print(f"baseline agree@{P} = {base.mean():.2%}\n")

    # most-missed token set, fit on TRAIN only (same calibration the shipped lever uses)
    miss_tr = dense[tr][required_p(H[tr] @ Cnorm, true_clu[tr]) > P]
    vals, cnts = np.unique(miss_tr, return_counts=True)
    ranked = vals[np.argsort(-cnts)]              # most-frequently-missed first

    miss_mask = ~base                             # eval positions currently missed
    n_miss = int(miss_mask.sum())
    dense_te = dense[te]
    idx = np.arange(len(te))

    print(f"{n_miss} eval misses to rescue. Two mechanisms, same N-token target set:\n")
    print(f"  {'N':>5} | {'always-score':^25} | {'targeted second home':^25}")
    print(f"  {'':>5} | {'rescue':>7} {'agree':>7} {'rows/step':>9} | "
          f"{'rescue':>7} {'agree':>7} {'rows/step':>9}")
    for N in (64, 256, 1024, 4096):
        sel = ranked[:N]
        c2_arr = second_homes(sel, W, Cnorm, tok2clu)   # one second cluster per selected token

        # always-score: any missed position whose token is in the set is caught, flat N rows/step.
        in_set = np.isin(dense_te, sel)
        a_rescue = (miss_mask & in_set).sum() / max(n_miss, 1)
        a_agree = base.mean() + (~base).mean() * a_rescue

        # targeted second home: token reachable iff its second cluster is in the eval row's top-P.
        home2 = np.full(V, -1, np.int64)
        home2[sel] = c2_arr
        c2_te = home2[dense_te]                          # -1 where token not in set
        has = c2_te >= 0
        reach = np.zeros(len(te), bool)
        reach[has] = sims_te[idx[has], c2_te[has]] >= thresh[has]   # second cluster probed this step?
        s_rescue = (miss_mask & reach).sum() / max(n_miss, 1)
        s_agree = (base | reach).mean()
        # cost: extra rows/step = set-tokens whose second cluster is in the top-P, averaged over steps
        s_rows = (sims_te[:, c2_arr] >= thresh[:, None]).sum(1).mean()

        print(f"  {N:>5} | {a_rescue:>6.1%} {a_agree:>6.2%} {N:>9d} | "
              f"{s_rescue:>6.1%} {s_agree:>6.2%} {s_rows:>9.1f}")


if __name__ == "__main__":
    main()
