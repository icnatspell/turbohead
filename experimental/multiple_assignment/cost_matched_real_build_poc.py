r"""POC: COST-MATCHED r=2 with a REAL rebuilt partition (not the next-nearest hack).

WHY THIS, ON TOP OF multiple_assignment_poc.py
----------------------------------------------
The sibling PoC already showed the cost-matched point: r=2 @P=128 (4096 candidate rows) beats
r=1 @P=256 (also 4096 rows). But it cheated on the build — it kept the SHIPPED r=1 centroids and
just gave each token its next-nearest cluster as a 2nd home. A real r=2 deploy rebuilds the
centroids over the doubled memberships, which pulls in two directions:

  (+) soft homes give a tail token a second route -> recall up
  (-) every centroid is now a mean over ~2*cap members, so it BLURS -> routing for ALL tokens shifts

The sibling PoC only sees the (+). This one rebuilds the centroids so it sees (+) and (-) together.
Question: does the cost-matched win SURVIVE a real build, or does the blur eat it?

THE COMPARISON (all three gather the SAME 4096 candidate rows -> same stage-2 cost)
  A) r=1, shipped centroids, P=256      <- the deployed reference
  B) r=2, shipped centroids, P=128      <- sibling PoC's cost-matched (soft homes, sharp centroids)
  C) r=2, REBUILT centroids, P=128      <- the real build (soft homes, blurred centroids)

C >= B  -> rebuilding is safe; promote the real r=2 build.
C <  B  -> the blur hurts; ship soft homes on the shipped centroids instead (even simpler).

ponytail: recall CEILING only. home2 = next-best centroid (no balance pass), so per-cluster size is
~2*cap on average, not exactly 2*cap. Exact balance is a build detail for the equal-cap kernel; it
moves required-P at the 2nd decimal, not the verdict. Add the balanced 2-assignment pass at
graduation, not here.

Read-only research: imports core helpers, touches nothing in turbohead/.
Run: uv run python experimental/multiple_assignment/cost_matched_real_build_poc.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbohead.eval.agreement import collect_hidden, wikitext
from turbohead.surgery.build_clusters import assign_scores

MODEL = "Qwen/Qwen3-0.6B"
HEAD = "artifacts/qwen3_0_6b/head_W.npy"
NPZ = "artifacts/qwen3_0_6b/clusters.npz"
MAX_TOKENS = 4000
CHUNK = 4096


def second_home(W, C, home1):
    """Next-best centroid per token (the deploy rule: a token's 2 nearest centroids), excluding home1.
    Uses the same argmin-||x-c|| score as the balanced builder."""
    V = W.shape[0]
    cnorm2 = (C * C).sum(1)
    h2 = np.empty(V, np.int64)
    for i in range(0, V, CHUNK):
        sc = assign_scores(W[i:i + CHUNK], C, cnorm2)        # (blk,K)
        rows = np.arange(sc.shape[0])
        sc[rows, home1[i:i + CHUNK]] = -np.inf               # drop the primary
        h2[i:i + CHUNK] = sc.argmax(1)
    return h2


def rebuild_centroids(W, home1, home2, K):
    """Mean over each cluster's r=2 members (every token counts toward both its homes)."""
    D = W.shape[1]
    s = np.zeros((K, D), np.float64)
    np.add.at(s, home1, W)
    np.add.at(s, home2, W)
    cnt = np.bincount(home1, minlength=K) + np.bincount(home2, minlength=K)
    return (s / cnt[:, None]).astype(np.float32)


def cnorm_of(C):
    return (C / np.linalg.norm(C, axis=1, keepdims=True)).T.astype(np.float32)   # (D,K)


def agree_at(sims, homes, P):
    """homes: (T,) or (T,r) home clusters per position. Reachable at P if any home's rank <= P."""
    if homes.ndim == 1:
        homes = homes[:, None]
    best = np.take_along_axis(sims, homes, axis=1).max(1)     # best-routed home's score
    rank = (sims > best[:, None]).sum(1) + 1                  # its rank in the routing order
    return (rank <= P).mean()


def main():
    W = np.load(HEAD).astype(np.float32)
    npz = np.load(NPZ)
    Cn1 = npz["Cnorm"].astype(np.float32)        # (D,K) shipped centroids, normalized
    Vmap = npz["Vmap"]
    V, D = W.shape
    K, cap = Vmap.shape
    tok2clu = np.empty(V, np.int64)
    tok2clu[Vmap.reshape(-1)] = np.repeat(np.arange(K), cap)   # token -> its shipped cluster
    C1 = Cn1.T                                                  # (K,D) for scoring 2nd home

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    H, dense = collect_hidden(model, tok, wikitext(40000), max_tokens=MAX_TOKENS)
    T = H.shape[0]

    # r=2 memberships, then rebuilt centroids
    home1 = tok2clu                                            # (V,) shipped primary
    home2 = second_home(W, C1, home1)                          # (V,) next-best centroid
    C2 = rebuild_centroids(W, home1, home2, K)
    Cn2 = cnorm_of(C2)

    sims1 = H @ Cn1                                            # routing on shipped centroids
    sims2 = H @ Cn2                                            # routing on rebuilt centroids
    homes_pos = np.stack([home1[dense], home2[dense]], 1)      # (T,2) the position's two homes

    overlap = (home2 == home1).mean()    # should be ~0 (we excluded home1); sanity
    avg_blur = np.linalg.norm(C2 - C1, axis=1).mean()
    print(f"{T} eval positions | K={K} cap={cap} | home2==home1: {overlap:.3%} | "
          f"mean centroid shift after rebuild: {avg_blur:.3f}\n")
    print("all three gather 4096 candidate rows -> identical stage-2 cost:\n")

    a = agree_at(sims1, home1[dense], 256)
    b = agree_at(sims1, homes_pos, 128)
    c = agree_at(sims2, homes_pos, 128)
    print(f"  A  r=1  shipped centroids  @P=256  -> {a:.2%}   (deployed reference)")
    print(f"  B  r=2  shipped centroids  @P=128  -> {b:.2%}   (soft homes, sharp centroids)")
    print(f"  C  r=2  REBUILT centroids  @P=128  -> {c:.2%}   (the real build)")
    print(f"\n  verdict: C-vs-B = {(c - b) * 100:+.2f}pp  |  C-vs-A = {(c - a) * 100:+.2f}pp")
    print("  C>=B: rebuild safe, promote real r=2.  C<B: blur hurts, ship soft homes on shipped centroids.")


if __name__ == "__main__":
    main()
