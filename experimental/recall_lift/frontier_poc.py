"""POC: does always-score (lever 4) let us run a LOWER P for the same recall — a TPS win?

The strategy question from the routing-matrix dead-ends: all the recall wins now keep routing exact;
the cheapest reachable one is always-score (a fixed list of chronic-misrouter tokens scored every step
via the Wspec path, ~free). It rescues P-INDEPENDENT misses (frequent tokens with a bad cluster
assignment, missed at any P). P rescues a DIFFERENT population (the moderate-rank tail). So always-score
cannot substitute for P — but it shifts the recall-vs-P curve UP by a roughly constant amount, which
means you can pick a smaller P for a target recall. This measures that frontier shift exactly.

Agreement == stage-1 recall: a position is correct if its true cluster ranks <= P in cosine order
(stage 2 is exact) OR its dense-argmax token is in the always-score set. We sweep P with the list OFF
(N=0) and ON (N=64), on a HELD-OUT split, with the list calibrated on the train half AT EACH P (the
list you'd actually ship for that P). Output: the lift at each P (is it constant?), and the matched-
recall P — the smallest P that, with always-score on, matches the P=256 no-list baseline. The TPS that
P reduction is worth gets priced by a real re-splice bench separately (see the README).

Reuses the cached hidden states from learned_metric (same model/slices). Read-only research.
Run: uv run python experimental/recall_lift/frontier_poc.py
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import wikitext

MODEL = "Qwen/Qwen3-0.6B"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
CACHE = "/tmp/learned_metric_HD.npz"          # shared with learned_metric: H (lm_head in) + dense argmax
CHUNK, N_CHUNKS, SEED = 2000, 6, 0
P_GRID = [32, 48, 64, 96, 128, 192, 256, 384, 512]
N_LIST = [0, 64, 128]                          # always-score list sizes (0 = off); 128 to check plateau
CALIB_P = 256                                  # P at which the always-score list is calibrated (shipped default)


def collect():
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


def always_set(dense_tr, rank_tr, N):
    """The shipped lever-4 list: the N most FREQUENT train tokens that route badly (true cluster ranks
    below CALIB_P), so they're scored every step regardless of P. Calibrated on train only."""
    if N == 0:
        return np.empty(0, np.int64)
    missed = dense_tr[rank_tr > CALIB_P]
    vals, cnts = np.unique(missed, return_counts=True)
    return vals[np.argsort(-cnts)][:N]


def main():
    rng = np.random.default_rng(SEED)
    npz = np.load(NPZ)
    Cnorm = npz["Cnorm"].astype(np.float32)
    Vmap = npz["Vmap"]
    K, cap = Vmap.shape
    tok2clu = np.empty(K * cap, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)

    H, dense = collect()
    true_clu = tok2clu[dense]
    T = H.shape[0]
    perm = rng.permutation(T)
    tr, te = perm[: T // 2], perm[T // 2:]
    rank_tr = required_p(H[tr] @ Cnorm, true_clu[tr])
    rank_te = required_p(H[te] @ Cnorm, true_clu[te])
    dense_te = dense[te]
    print(f"{T} positions ({len(tr)} fit / {len(te)} eval) | K={K} cap={cap} | calib P={CALIB_P}\n")

    # agree(P, N) on eval = reached by probing P  OR  token in the (train-fit) always-score set
    sets = {N: set(always_set(dense[tr], rank_tr, N).tolist()) for N in N_LIST}
    in_set = {N: np.array([t in sets[N] for t in dense_te]) for N in N_LIST}

    def agree(P, N):
        return ((rank_te <= P) | in_set[N]).mean()

    hdr = "  P    " + "".join(f"N={N:<8}" for N in N_LIST) + "  lift(N=64)"
    print("AGREEMENT vs P (rows) x always-score list size (cols):")
    print(hdr)
    for P in P_GRID:
        cells = "".join(f"{agree(P, N):<10.2%}" for N in N_LIST)
        lift = agree(P, 64) - agree(P, 0)
        print(f"  {P:<5}{cells}  +{lift:.2%}")

    # matched-recall: smallest P that with always-score on reaches the P=256, list-off baseline
    base = agree(256, 0)
    print(f"\nbaseline (P=256, no list): {base:.2%}")
    for N in (64, 128):
        hits = [P for P in P_GRID if agree(P, N) >= base]
        if hits:
            Pm = min(hits)
            print(f"  with always-score N={N}: P={Pm} already matches it "
                  f"(agree {agree(Pm, N):.2%}) -> probe {256}->{Pm} = {1 - Pm / 256:.0%} fewer stage-2 rows")
        else:
            print(f"  with always-score N={N}: no P in grid matches the baseline")

    # how does the lift scale with P? it GROWS as P shrinks (more of the shrinking-P miss pool is the
    # fixed chronic-misrouter set), so the low-P + always-score strategy is stronger than a constant.
    lifts = [(P, agree(P, 64) - agree(P, 0)) for P in P_GRID]
    lo, hi = lifts[-1], lifts[0]
    print(f"\nalways-score lift grows as P shrinks: +{hi[1]:.2%} @P={hi[0]} -> +{lo[1]:.2%} @P={lo[0]} "
          f"(biggest exactly where you want to operate, not a flat P-independent offset)")


if __name__ == "__main__":
    main()
