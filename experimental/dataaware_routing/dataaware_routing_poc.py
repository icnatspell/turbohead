"""POC for thesis dir #4 (data-aware clustering), cheapest variant: data-aware ROUTING.

Keep the existing cluster groups (which tokens are in which cluster) unchanged. Only change
the stage-1 routing centroid:
  baseline   : centroid_k = mean of token EMBEDDINGS in cluster k   (current FlashHead)
  data-aware : centroid_k = mean of HIDDEN STATES whose true token is in cluster k (query space)

Metric = required-P (rank of the true cluster in the centroid-score order). Lower is better.
Fit centroids on the first half of positions, evaluate on the second half (no leakage).
If data-aware routing lowers required-P, #4 has value AND it deploys by swapping Cnorm only.
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

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
H, dense = collect_hidden(model, tok, wikitext(80000), max_tokens=8000)   # H:(T,D), dense:(T,)
T = H.shape[0]

tok2clu = np.empty(K * cap, np.int64)
for k in range(K):
    tok2clu[Vmap[k]] = k
true_clu = tok2clu[dense]                       # (T,) cluster of the true token

half = T // 2
fit, ev = slice(0, half), slice(half, T)
print(f"{T} positions | fit={half} eval={T-half} | K={K} cap={cap}")

def required_p(cent, Hs, tclu):
    """cent:(D,K) unit-norm routing centroids. Return required-P array for positions Hs/tclu."""
    sims = Hs @ cent                            # (n, K)
    ts = sims[np.arange(len(Hs)), tclu]
    return (sims > ts[:, None]).sum(1) + 1

def report(name, rank):
    pct = {p: int(np.percentile(rank, p)) for p in (50, 90, 95, 99)}
    accs = {P: f"{(rank <= P).mean():.1%}" for P in (64, 256)}
    print(f"  {name:12s} p50={pct[50]:<4} p90={pct[90]:<5} p95={pct[95]:<5} p99={pct[99]:<6}"
          f" mean={rank.mean():6.1f} | acc@64={accs[64]} acc@256={accs[256]}")

# baseline: embedding-mean centroids (already in the npz, already unit-norm)
report("baseline", required_p(Cnorm, H[ev], true_clu[ev]))

# query-space centroids fit on the first half
q = np.zeros((D, K), np.float32)
np.add.at(q.T, true_clu[fit], H[fit])          # sum hidden states per cluster
cnt = np.bincount(true_clu[fit], minlength=K)
q[:, cnt > 0] /= np.linalg.norm(q[:, cnt > 0], axis=0, keepdims=True) + 1e-9

# CLEAN sub-problem: only well-populated clusters, rank within that consistent set, eval on
# positions whose true cluster is populated. Same candidate set for both methods -> fair.
MIN = 3
seen = np.where(cnt >= MIN)[0]
ev_mask = np.isin(true_clu[ev], seen)
He, te = H[ev][ev_mask], true_clu[ev][ev_mask]
local = {c: i for i, c in enumerate(seen)}      # global cluster id -> column in the seen set
te_local = np.array([local[c] for c in te])
print(f"  clean sub-problem: {len(seen)} clusters (>={MIN} fit samples), {len(He)} eval positions")

def req_p_subset(cent_cols):
    sims = He @ cent_cols                        # (n, |seen|)
    ts = sims[np.arange(len(He)), te_local]
    return (sims > ts[:, None]).sum(1) + 1

report("baseline*", req_p_subset(Cnorm[:, seen]))   # embedding centroids, seen set only
report("query*",    req_p_subset(q[:, seen]))       # query centroids, seen set only
report("blend*", req_p_subset(
    (lambda b: b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-9))(
        0.5 * Cnorm[:, seen] + 0.5 * q[:, seen])))
