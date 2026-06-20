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

## Index

| folder | what it explored | verdict |
|---|---|---|
| `recall_lift/` | four ways to raise top-1 agreement (better routing matrix, exact-stop, margin cascade, always-score) | **shipped** the always-score win (`turbohead-calibrate-misses`); 1–3 parked |
| `buffershare/` | in-place buffer-shared KV cache via IOBinding | **shipped** (`decode_loop.py` `share_kv`) |
| `coverage_correction/` | corrected softmax denominator for honest PPL when the token is known | win on likelihood; promote pending |
| `adaptive_probe_headroom/` | per-token required-P distribution (the adaptive-probing ceiling) | analysis behind `docs/THESIS_ADAPTIVE_PROBING.md` |
| `mips_routing/` | rank clusters by inner product / norm-bound instead of cosine | parked — cosine already wins |
| `hierarchical_stage1/` | coarse-to-fine (super-centroid) routing to cut the stage-1 floor | parked — coarse prune drops the hard tail |
| `dataaware_routing/` | fit the routing matrix to data instead of embedding means | parked — no beat over cosine |
| `graph_mips/` | FAISS HNSW graph search as the router | reference (see `docs/RELATED.md`) |
| `l2s/` | learning-to-search framing | reference (see `docs/RELATED.md`) |
| `genai_kv/`, `kv_scaling/` | KV-cache A/Bs behind the buffer-share work | analysis (see `docs/IDEAS.md`, KV section) |

See `docs/IDEAS.md` for the full write-ups and numbers; this folder holds the runnable scripts.
