# experimental/

Throwaway scripts for ideas we tried on top of the core method — kept as the evidence behind
what shipped and what got parked. **Nothing in `turbohead/` or `csrc/` depends on anything here.**

## Convention

- **One idea per folder.** Each folder is standalone — run it directly (`uv run python
  experimental/<name>/<file>.py`).
- **Core stays untouched.** An experiment may *import* core helpers read-only (e.g.
  `from turbohead.eval.agreement import collect_hidden`), but anything it changes about the method
  it overrides *inside its own file* — never by editing `turbohead/`. So the core is always the
  shipping truth and an experiment can diverge freely.
- **Promotion.** When an experiment proves out, lift its override into the corresponding core module
  (and delete or annotate the experiment). That is how `buffershare/` and `recall_lift/` graduated.
  `anisotropic_clustering/` graduated as an opt-in `--eta` knob (not a default — see its row below).

## Index

| folder | what it explored | verdict |
|---|---|---|
| `multiple_assignment/` | give each token its top-r home clusters (overlapping IVF) so the tail is reachable | **win on quality** — r=2 → 99.0% agree@256, but costs ~25% decode (stage 2 is ~28% of the step) |
| `targeted_second_home/` | second home only for the heavy-miss tokens — bake-off vs shipped always-score | parked — dominated; captures 84% of always-score's lift but always-score is already ~free and scores higher |
| `anisotropic_clustering/` | ScaNN-style MIPS-aware partition (penalise parallel quantization error) | **shipped as opt-in** — `build_clusters.py --eta` (default 1.0); PER-MODEL: eta=4 helps Qwen3-0.6B, regresses gemma3-270m |
| `whitened_routing/` | learned Mahalanobis routing metric folded into the centroids | parked — whitening hurts; LLM anisotropy is signal, not noise |
| `factorized_router/` | product-quantized (cheap) router to afford a bigger P | parked — too lossy; can't beat exact even at 4× P |
| `combinations/` | do levers stack or overlap? (first study: anisotropic_clustering × multiple_assignment) | aniso×multi do **not** stack (sub-additive); both fix the same tail, so ship one |
| `recall_lift/` | four ways to raise top-1 agreement (better routing matrix, exact-stop, margin cascade, always-score) | **shipped** the always-score win (`turbohead-calibrate-misses`); 1–3 parked |
| `buffershare/` | in-place buffer-shared KV cache via IOBinding | **shipped** (`decode_loop.py` `share_kv`) |
| `coverage_correction/` | corrected softmax denominator for honest PPL when the token is known | parked — the honest report (covered PPL + coverage%) already ships in `eval/{ppl,head_quality}`; the corrected-Z denominator is a 0.01 PPL no-op |
| `adaptive_probe_headroom/` | per-token required-P distribution (the adaptive-probing ceiling) | analysis behind `docs/THESIS_ADAPTIVE_PROBING.md` |
| `mips_routing/` | rank clusters by inner product / norm-bound instead of cosine | parked — cosine already wins |
| `hierarchical_stage1/` | coarse-to-fine (super-centroid) routing to cut the stage-1 floor | parked — coarse prune drops the hard tail |
| `dataaware_routing/` | fit the routing matrix to data instead of embedding means | parked — no beat over cosine |
| `graph_mips/` | FAISS HNSW graph search as the router | reference (see `docs/RELATED.md`) |
| `l2s/` | learning-to-search framing | reference (see `docs/RELATED.md`) |
| `genai_kv/`, `kv_scaling/` | KV-cache A/Bs behind the buffer-share work | analysis (see `docs/IDEAS.md`, KV section) |

## By goal (what each experiment attacks)

The index above is sorted by verdict. This view groups the same folders by *what they try to move*.
FlashHead has two stages: **stage 1** routes the hidden state to the top-P clusters (a `[K,D]` gemv +
TopK), **stage 2** gathers and exactly scores the `P·cap` candidate rows. Top-1 agreement is decided
entirely by stage-1 recall (did the true token's cluster make the top-P?), so most quality work lives
there. A few experiments sit off the head entirely.

### Stage-1 recall — reshape the partition (better clusters, same inference cost)

| folder | idea | verdict |
|---|---|---|
| [`anisotropic_clustering/`](anisotropic_clustering/) | penalise the *parallel* quantization error (ScaNN), the part that moves an inner product | **shipped as opt-in `--eta`** (default 1.0); per-model — helps Qwen3-0.6B, regresses gemma3-270m |
| [`whitened_routing/`](whitened_routing/) | learned Mahalanobis metric folded into the centroids | parked — whitening hurts; the anisotropy is signal |
| [`dataaware_routing/`](dataaware_routing/) | fit the routing matrix to data instead of embedding means | parked — no beat over cosine |

### Stage-1 recall — change the routing score (same partition, rank clusters differently)

| folder | idea | verdict |
|---|---|---|
| [`mips_routing/`](mips_routing/) | rank by inner product / norm-bound instead of cosine | parked — cosine already wins |
| [`recall_lift/`](recall_lift/) (levers 2–3) | exact-stop certificate; margin-gated cascade re-probe | parked — exact-stop degenerates to full vocab; cascade ≈ break-even with raising P |

### Stage-1 recall — add reachability (catch the heavy miss tail)

| folder | idea | verdict |
|---|---|---|
| [`recall_lift/`](recall_lift/) (lever 4) | always-score the chronically-missed tokens, bypassing routing | **shipped** — `turbohead-calibrate-misses`; +1.3pp agree@256 at ~free cost |
| [`targeted_second_home/`](targeted_second_home/) | second home for the same heavy-miss tokens — reachable via a 2nd route, not scored every step | parked — dominated by always-score (84% of the lift, but always-score is already ~free and scores higher) |
| [`multiple_assignment/`](multiple_assignment/) | give each token its top-r home clusters (overlapping IVF) | **win on quality** — r=2 → 99.0% agree@256, but **costs stage-2 speed** (~25% decode) |

### Stage-1 speed — lower the routing floor (cheaper/coarser router to afford a bigger P)

| folder | idea | verdict |
|---|---|---|
| [`hierarchical_stage1/`](hierarchical_stage1/) | coarse-to-fine super-centroid routing | parked — coarse prune drops the hard tail; only helps at very large K |
| [`factorized_router/`](factorized_router/) | product-quantized (cheap) router | parked — too lossy; can't beat exact even at 4× P |

### Adaptive per-token compute (spend probe budget where it's needed)

| folder | idea | verdict |
|---|---|---|
| [`adaptive_probe_headroom/`](adaptive_probe_headroom/) | per-token required-P distribution: the adaptive-probing ceiling | analysis behind `docs/THESIS_ADAPTIVE_PROBING.md`; no cheap confidence signal found |

### Fidelity, not speed or agreement (orthogonal)

| folder | idea | verdict |
|---|---|---|
| [`coverage_correction/`](coverage_correction/) | corrected softmax denominator for honest PPL when the token is known | parked — honest report already ships (`eval/{ppl,head_quality}` print covered PPL + coverage%); corrected-Z is a 0.01 PPL no-op |

### Off the head — the body / KV cache (orthogonal to FlashHead)

| folder | idea | verdict |
|---|---|---|
| [`buffershare/`](buffershare/) | in-place buffer-shared KV cache via IOBinding | **shipped** (`decode_loop.py` `share_kv`) |
| [`genai_kv/`](genai_kv/), [`kv_scaling/`](kv_scaling/) | KV-cache A/Bs behind the buffer-share work | analysis (see `docs/IDEAS.md`, KV section) |

### Meta — do levers stack or overlap?

| folder | idea | verdict |
|---|---|---|
| [`combinations/`](combinations/) | test pairs of levers for additivity | aniso × multi do **not** stack (both fix the same tail) |

### Reference — external methods, not run as PoCs

| folder | idea | verdict |
|---|---|---|
| [`graph_mips/`](graph_mips/) | FAISS HNSW graph search as the router | reference (see `docs/RELATED.md`) |
| [`l2s/`](l2s/) | learning-to-search framing | reference (see `docs/RELATED.md`) |

See `docs/IDEAS.md` for the full write-ups and numbers; this folder holds the runnable scripts.
