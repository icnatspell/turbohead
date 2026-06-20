"""POC: can we lift top-1 agreement WITHOUT raising the probe budget P?

Agreement == stage-1 recall: FlashHead returns the dense argmax token exactly when that token's
cluster lands in the top-P clusters by stage-1 score (stage 2 is exact). So "required-P" of a
position = the rank of its true cluster in the stage-1 ordering; agreement@P = fraction with
required-P <= P. Everything below just recomputes required-P under different stage-1 ideas, on a
HELD-OUT split, against the same fixed clusters (stage 2 never changes).

Two levers, both aimed at "don't slow down at all":

  Lever 1 — better routing MATRIX (zero runtime cost: same K x D gemv, just different numbers).
    Current routing is cosine vs the mean EMBEDDING of each cluster. We try routing vectors fit to
    the hidden states that actually select each cluster, plus two free linear re-framings of h
    (mean-centering, diagonal whitening) that counter the anisotropy of LLM hidden states.

  Lever 2 — bound-guided EXACT-STOP probing (adaptive P; aims for 100% agreement, asks what avg P
    that costs). Probe clusters in cosine order, tracking the best exact logit L* found so far.
    A cluster k can only beat L* if its provable upper bound  mu_k.h + ||h||*radius_k  exceeds L*
    (Cauchy-Schwarz: every member sits within radius_k of the mean). Once every UNPROBED cluster's
    bound is below L*, no unscored token can win -> L* is the true global max -> exact top-1, and we
    stop. This is guaranteed-correct by construction; the empirical question is the probe count it
    forces (avg and tail). Low avg => 100% agreement at ~current cost. High avg => too slow.

Read-only research: touches nothing in the shipping path. Run: uv run python logs/recall_lift_poc.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import wikitext

MODEL = "Qwen/Qwen3-0.6B"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
CHUNK = 2000           # tokens per forward pass (kept small so the (T,V) logits stay cheap)
N_CHUNKS = 6           # disjoint text slices -> ~12k positions total; split 50/50 fit / eval
SEED = 0


def collect(model, tok, texts):
    """Capture, per position: h (input to lm_head), the dense argmax token, and the best EXACT
    logit inside each cluster (max over its cap members). bestlog drives lever 2's stop bound.
    Runs each text slice as its own forward (real context, but bounded seq length) and concatenates."""
    Vmap = np.load(NPZ)["Vmap"]                          # (K, cap)
    flat = Vmap.reshape(-1)
    K, cap = Vmap.shape
    Hs, ds, bs = [], [], []
    for text in texts:
        h_in = {}
        hook = model.lm_head.register_forward_hook(lambda m, i, o: h_in.update(h=i[0]))
        ids = tok(text, return_tensors="pt").input_ids[:, :CHUNK]
        with torch.no_grad():
            logits = model(ids).logits[0].float().numpy()   # (T, V) — transient, reduced below
        hook.remove()
        H = h_in["h"][0].float().numpy()                 # (T, D)
        bestlog = logits[:, flat].reshape(-1, K, cap).max(2)   # (T,K): best exact logit per cluster
        Hs.append(H)
        ds.append(logits.argmax(-1))
        bs.append(bestlog)
    return np.concatenate(Hs), np.concatenate(ds), np.concatenate(bs)


def required_p(sims, true_clu):
    """Rank of each position's true cluster under routing scores `sims` (T,K). 1 = best. Lower=better."""
    ts = sims[np.arange(sims.shape[0]), true_clu]
    return (sims > ts[:, None]).sum(1) + 1


def report(name, rank, runtime_note):
    pct = {p: int(np.percentile(rank, p)) for p in (50, 90, 99)}
    acc = {P: f"{(rank <= P).mean():.2%}" for P in (128, 256, 512)}
    print(f"  {name:16s} p50={pct[50]:<3} p90={pct[90]:<4} p99={pct[99]:<6} "
          f"| agree @128={acc[128]} @256={acc[256]} @512={acc[512]}   [{runtime_note}]")


def main():
    rng = np.random.default_rng(SEED)
    npz = np.load(NPZ)
    Cnorm = npz["Cnorm"].astype(np.float32)              # (D,K) unit mean-embedding directions
    Wperm = npz["Wperm"].astype(np.float32)              # (K,cap,D)
    Vmap = npz["Vmap"]
    D, K = Cnorm.shape
    cap = Wperm.shape[1]

    mu = Wperm.mean(1)                                    # (K,D) raw mean embedding
    rad = np.linalg.norm(Wperm - mu[:, None, :], axis=2).max(1)   # (K,) cluster radius (exact bound)
    tok2clu = np.empty(K * cap, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    big = wikitext(CHUNK * N_CHUNKS * 12)                 # plenty of chars; slice into N_CHUNKS pieces
    step = len(big) // N_CHUNKS
    H, dense, bestlog = collect(model, tok, [big[i * step:(i + 1) * step] for i in range(N_CHUNKS)])
    true_clu = tok2clu[dense]
    T = H.shape[0]

    # held-out split: fit lever-1 routing on `tr`, evaluate required-P on `te` (no peeking)
    perm = rng.permutation(T)
    tr, te = perm[: T // 2], perm[T // 2:]
    Hte, cte = H[te], true_clu[te]
    sims_te = Hte @ Cnorm                                 # cosine routing scores, reused by all levers
    print(f"{T} positions ({len(tr)} fit / {len(te)} eval) | K={K} cap={cap} D={D}\n")

    # ---- Lever 1: routing-matrix swaps (all zero extra runtime: one K x D gemv either way) ----
    print("LEVER 1  routing matrix (held-out required-P; lower is better):")
    report("cosine (current)", required_p(sims_te, cte), "baseline")

    # (a) prototypes = mean of the TRAIN hidden states that select each cluster, cosine-normalized.
    #     Clusters never selected in train (most of them: K>>T) fall back to the embedding direction.
    sums = np.zeros((K, D), np.float32)
    np.add.at(sums, true_clu[tr], H[tr])
    cnt = np.bincount(true_clu[tr], minlength=K)
    proto = Cnorm.T.copy()                                # default = current routing direction
    seen = cnt > 0
    proto[seen] = sums[seen] / cnt[seen, None]
    proto /= np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9
    cov = np.isin(cte, np.flatnonzero(seen)).mean()
    report("learned-h proto", required_p(Hte @ proto.T, cte),
           f"free; only {cov:.0%} of eval clusters had fit data")

    # (b) mean-center h (remove the common-mode direction LLM hidden states share). Equivalent to a
    #     fixed per-cluster bias on the cosine score -> still one gemv + an added constant vector.
    hbar = H[tr].mean(0)
    report("centered cosine", required_p((Hte - hbar) @ Cnorm, cte), "free (h-bias fold-in)")

    # (c) diagonal whitening: scale each dim by 1/std(h) before routing (counters outlier dims that
    #     dominate the dot product). Fold the scale into both h and the centroids offline -> free.
    inv = 1.0 / (H[tr].std(0) + 1e-6)
    muw = mu * inv
    muw /= np.linalg.norm(muw, axis=1, keepdims=True) + 1e-9
    report("whitened cosine", required_p(((Hte - hbar) * inv) @ muw.T, cte), "free (diag fold-in)")

    # ---- Lever 2: bound-guided exact-stop (adaptive P; guaranteed-correct; measure the cost) ----
    print("\nLEVER 2  bound-guided exact-stop (uncapped = 100% agreement by construction):")
    hnorm = np.linalg.norm(Hte, axis=1)
    raw_ip = Hte @ mu.T                                   # (Te,K) mu.h, hoisted out of the loop
    blog_te = bestlog[te]
    stopP = np.empty(len(te), np.int64)
    for i in range(len(te)):
        order = np.argsort(-sims_te[i])                   # probe clusters best-cosine first
        bound = raw_ip[i][order] + hnorm[i] * rad[order]  # upper bound per cluster, in probe order
        Lstar = np.maximum.accumulate(blog_te[i][order])  # best exact logit after probing 1..p
        suffix_max_bound = np.maximum.accumulate(bound[::-1])[::-1]   # max bound among rank>=p
        # stop at smallest p where every cluster AFTER p has bound < L* found in 1..p
        safe = suffix_max_bound[1:] < Lstar[:-1]
        stopP[i] = (np.argmax(safe) + 1) if safe.any() else K
    p = stopP
    print(f"  exact-stop P: mean={p.mean():.1f}  p50={int(np.percentile(p,50))}  "
          f"p90={int(np.percentile(p,90))}  p99={int(np.percentile(p,99))}  max={p.max()}")
    print(f"  -> 100% agreement; avg probes {p.mean():.0f} vs the fixed P=256 the deploy uses now.")
    for C in (256, 512, 1024):
        print(f"     fraction needing P> {C:<4d}: {(p > C).mean():.2%}")

    # ---- Lever 3: margin-gated cascade (IDEAS #5) — probe P0, escalate to P1 only when the top-2
    #      probed exact logits are close (a cheap "this might be wrong" signal). Heuristic, not a
    #      guarantee: measures whether the confidence margin actually predicts the misses. Avg probe
    #      cost is between P0 and P1; we want high agreement at low avg. ----
    print("\nLEVER 3  margin-gated cascade (escalate the uncertain few):")
    rank_te = required_p(sims_te, cte)                    # cosine required-P on eval
    P0, P1 = 64, 512
    order_te = np.argsort(-sims_te, axis=1)               # (Te,K) clusters best-cosine first
    top2 = np.sort(np.take_along_axis(blog_te, order_te[:, :P0], axis=1), axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]                      # gap between best & 2nd-best probed logit
    base0, base1 = (rank_te <= P0), (rank_te <= P1)
    print(f"  fixed baselines: P0={P0} agree={base0.mean():.2%} | P1={P1} agree={base1.mean():.2%}")
    print(f"  {'margin<thr':>11} {'escalated':>10} {'avg P':>7} {'agreement':>10}")
    for thr in (0.0, 0.5, 1.0, 2.0, 4.0):
        esc = margin < thr                               # escalate these positions to P1
        agree = (base0 & ~esc) | (base1 & esc)           # correct if reached within the P actually spent
        avgP = np.where(esc, P1, P0).mean()
        print(f"  {thr:>11.1f} {esc.mean():>10.1%} {avgP:>7.0f} {agree.mean():>10.2%}")

    # ---- Lever 4: always-score a fixed token set (reuses the graph's existing Wspec/EOS path —
    #      a handful of extra rows scored every step, ~free). Pays off only if the misses concentrate
    #      on few tokens. Fit the "most-missed at P=256" set on train, measure the lift on eval. ----
    print("\nLEVER 4  always-score the most-missed tokens (fixed set, ~free via Wspec path):")
    P = 256
    miss_tr = dense[tr][required_p(H[tr] @ Cnorm, true_clu[tr]) > P]   # tokens flash misses on train
    vals, cnts = np.unique(miss_tr, return_counts=True)
    ranked = vals[np.argsort(-cnts)]                     # most-frequently-missed first
    base = (rank_te <= P)
    miss_te_tok = dense[te][~base]                       # the eval tokens currently missed
    print(f"  baseline agree@{P}={base.mean():.2%}; {len(np.unique(miss_te_tok))} distinct missed "
          f"tokens on eval, {len(miss_te_tok)} misses")
    for N in (64, 256, 1024, 4096):
        always = set(ranked[:N].tolist())
        rescued = np.array([t in always for t in miss_te_tok]).mean() if len(miss_te_tok) else 0.0
        lifted = base.mean() + (~base).mean() * rescued
        print(f"  always-score top {N:>4d}: rescues {rescued:>5.1%} of misses -> agree {lifted:.2%}")


if __name__ == "__main__":
    main()
