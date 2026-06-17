# FlashHead via ONNX Graph Surgery — Implementation Plan

> **Status:** this is the original design/handoff spec; its `§N` section numbers are referenced from
> code comments, so it's kept as-is. For **what was built and measured**, see
> [RESULTS.md](RESULTS.md) (8-model matrix); for **current state + open items**, [NEXT_STEPS.md](NEXT_STEPS.md).

**Goal:** Insert an approximate, clustering-based LM head ("FlashHead") into an **existing, already-quantized ONNX language model** by graph surgery, replacing the dense final vocabulary projection on the decode path. Start with the simplest correct version (greedy, batch size 1, decode-only) and build up. The first priority is to confirm a real latency benefit at the target model's dimensions and execution provider, then preserve it through integration.

This document is a handoff spec for an implementing agent. It encodes the algorithm, the exact tensor layouts, the surgery procedure, validation gates, and the known gotchas. Reference: `embedl/flash-head` (vLLM plugin); arXiv 2603.14591. The vLLM plugin scaffolding (`patches/`, `__init__.py`, `loading.py`) does **not** transfer — only the algorithm and the offline clustering assets do.

---

## 0. Bootstrap target & measurement harness (Qwen3-0.6B, CPU)

Everything below in this section is the concrete starting point. Get this working and measured **before** building clustering or doing surgery — it establishes the baseline numbers, the harness, and whether the head is even worth attacking on this model.

### 0.1 Model and conversion

Target: **`Qwen/Qwen3-0.6B`**, converted to ONNX with the **onnxruntime-genai model builder**. Relevant config facts (verified from the HF `config.json`):

| | |
|---|---|
| `vocab_size` (V) | 151936 |
| `hidden_size` (D) | 1024 |
| `num_hidden_layers` | 28 |
| `head_dim` / heads / kv-heads | 128 / 16 / 8 (GQA) |
| `tie_word_embeddings` | **true** |
| `eos_token_id` | 151645 (also treat 151643 bos as special) |
| dtype | bfloat16 |

Two consequences that shape the whole effort:

- **`V = 151936 = 128 × 1187` (1187 is prime).** So `V` is divisible by every power of two up to 128. Pick `cap = 16 → K = 9496` (`= 8 × 1187`), or `cap = 32 → K = 4748`, etc. **No vocabulary padding is needed** — the balanced-cluster invariant `K·cap = V` is satisfiable exactly. This removes the §3 padding gotcha for the bootstrap.
- **Tied embeddings.** The LM head weight *is* the input-embedding weight. This matters in two places: (a) the quantization recipe (below) — embed inherits the head's int8 weights via weight sharing; (b) the surgery — **the original head/embed weight cannot be deleted** after FlashHead replaces the head MatMul, because the input embedding `Gather` still needs it. So the v1 "second copy of the head weights" (`Wperm`) is strictly additive here, and there is no size to reclaim by deleting the dense head. Budget for it.

### 0.2 Quantization recipe → exact build command

Recipe: **INT4 weights everywhere except LM-head and embeddings, which are INT8**, `accuracy_level=4`, group size `128`.

```bash
python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen3-0.6B \
  -o ./qwen3_0_6b_int4_cpu \
  -p int4 \
  -e cpu \
  -c ./hf_cache \
  --extra_options int4_block_size=128 int4_accuracy_level=4 int4_algo_config=k_quant_last
```

Flag rationale:
- `int4_block_size=128` — the group size for int4 (MatMulNBits) quantization.
- `int4_accuracy_level=4` — int8 compute path: activations are quantized to int8 and the int4 weights are upcast to int8 for the matmul. This is the requested accuracy level.
- `int4_algo_config=k_quant_last` — body MatMuls quantized to int4 with k-quant; **only the last MatMul (`/lm_head/MatMul`) is quantized to int8**. (`rtn_last` is the simpler RTN equivalent — try it as a fallback; k-quant generally gives better accuracy.)
- `tie_word_embeddings=true` ⇒ **shared embeddings auto-enable**, so the int8 head weights are shared with the embedding lookup — that is how "embed = INT8" is achieved without a separate flag.

**Verification step (do not skip).** After building, inspect the produced `model.onnx` node dtypes and confirm: (1) body projection MatMuls are 4-bit `MatMulNBits` with `block_size=128`; (2) the `/lm_head/MatMul` weight is 8-bit; (3) how the embedding is represented (a `GatherBlockQuantized` sharing the head weights vs a separate `Gather`). The int8-head-plus-sharing combination on a tied model is the part most likely to behave unexpectedly across builder versions — if sharing is disabled or the head lands at int4, adjust (`rtn_last`, or explicitly set precisions) and re-verify. For FlashHead itself the exact embed dtype barely matters (embed is a cheap `Gather`), but the **baseline head dtype must be known** for an honest A/B.

### 0.3 Why this model is a good first target (head-share estimate)

Back-of-envelope, weight-bandwidth-only (decode is bandwidth-bound on CPU): non-embedding params ≈ 596M − 155.6M ≈ 440M at int4 ≈ ~220 MB/token; the head ≈ V·D = 155.6M at int8 ≈ ~155 MB/token. So the **head is ~40% of per-token weight bytes**. If FlashHead cuts the head to ~9.4% of its work, per-token bytes drop ~376→235 MB, a **~1.5–1.6× decode-level ceiling** for this model — consistent with the reference's "up to 2×" and high enough to be worth it. Treat this only as the Amdahl ceiling; real gains will be lower after attention/KV and op overhead.

### 0.4 Measurement harness (CPU EP only)

All measurement on **`CPUExecutionProvider`**. Pin threads for reproducibility (`session intra_op_num_threads = fixed`, set `OMP_NUM_THREADS`); report core count, whether threads are pinned, and machine. Use **warmup + multiple timed repetitions**, report **median + IQR** (not just mean), discard warmup.

- **TTFT (time to first token)** — prefill latency to the first emitted token. Measure with the onnxruntime-genai `Generator` API. Sweep **prompt lengths** (e.g. ~16 / ~128 / ~512 / ~1024 tokens) since TTFT scales with prompt length.
- **TPS (decode tokens/sec)** — steady-state decode rate = `(new_tokens − 1) / (total_time − TTFT)` at a fixed `max_new_tokens` (e.g. 128–256). Several prompts; median across reps and prompts.
- **PPL (perplexity)** — accuracy/quality metric. Easiest reliable route: run the **bare ONNX model directly under ORT** (not the genai loop), feeding `input_ids` + empty/zero-length past + `position_ids` + `attention_mask` in a single forward per eval window, read `logits (1, L, V)`, compute teacher-forced NLL on shifted targets with a sliding window/stride over a standard corpus (e.g. WikiText-2 raw test). This sidesteps KV-cache bookkeeping and gives exact per-token logits.
- **Prompt set** — a small fixed suite spanning short instructions, multi-turn chat, long-context, and code, so TTFT/TPS aren't tuned to one shape. Keep it version-controlled and reused across phases for comparability.

> **PPL-under-FlashHead caveat (applies from Phase 2 on, not the baseline).** FlashHead only computes logits for the probed clusters, so a teacher-forced "FlashHead PPL" is only defined when the **ground-truth token's cluster is among the probes**. Report it as (a) **coverage** = fraction of eval tokens whose true cluster was probed, plus (b) NLL computed from the softmax over the *probed* token set (∪ always-scored specials), with a fixed floor when uncovered. This is **not** directly comparable to dense full-vocab PPL — compare FlashHead-PPL trends across `P`, and use coverage→1 as the target. Baseline (dense quantized head) PPL is standard full-vocab and is the honest reference point.

### 0.5 Bootstrap exit / go-no-go

Proceed past bootstrap only if all hold: the int4/int8 model builds and runs on CPU; baseline PPL is acceptable vs the bf16/HF reference (quantization didn't already break it); the harness produces stable TTFT/TPS (tight IQR); **the head's measured share of per-token decode time is large enough to matter** (profile the `/lm_head/MatMul` node via ORT profiling on a single decode step — if it is a few percent, FlashHead can't help much here); and the standalone `flashhead_probe.py` at the real dims (`V=151936, D=1024, cap=16, K=9496`, sweep `P`) shows a head-isolated speedup on CPU. If the head share is large and the probe is faster, continue to Phase 1.

---

## 1. Proposal

The standard LM head computes a dense `h_t · W_vocab` over all `V` tokens at every decode step. For small LLMs this is up to ~half of decode compute and is memory-bandwidth bound (it reads the entire `V×D` weight each step). FlashHead reframes it as two-stage retrieval over **balanced** clusters of token embeddings:

1. **Stage 1 (coarse):** score `K` cluster centroids, pick the top `P` ("probes").
2. **Stage 2 (fine):** score only the `P·cap` tokens inside those clusters, then argmax/sample.

Tokens scored drop from `V` to `K + P·cap`. With the reference's example (`V=128256`, `K=8016`, `cap=16`, `P≈256`) that is ~9.4% of the vocab. Because clustering is **balanced** (every cluster has exactly `cap` tokens, `K·cap = V`), all shapes are static — which is what makes a clean ONNX graph possible.

**Validated feasibility.** A standalone ONNX subgraph built from only `MatMul`, `TopK`, `Gather`, `ArgMax`, `Reshape`, `Transpose` (no custom ops) ran **~5.7× faster than a dense head** on single-thread CPU, fp32, at the 9.4%-scored ratio (proxy dims `V=32064, D=1024, K=2004, cap=16, P=64`). The accompanying `flashhead_probe.py` reproduces this; plug in the real dimensions/EP to re-confirm at scale before any integration work.

**Scope of v1 (this plan):**
- In scope: greedy decode, batch size 1, decode step only (prefill stays on the dense head), graph surgery into an existing quantized ONNX model.
- Out of scope for v1 (later phases): sampling with distribution preservation, speculative/EAGLE decoding, batch > 1, custom fused operators, edge-EP–specific tuning.

---

## 2. Algorithm, distilled from the reference code

From `flash_head.py`, the greedy / batch-1 / `T=1` path reduces to:

```
sims     = h @ Cnorm                 # (1, K)      Cnorm = unit centroids, columns
top      = TopK(sims, P).indices     # (1, P)      most-aligned clusters
rows     = Gather(Wperm, top)        # (P, cap, D) probed weight blocks
logits   = h @ rows.reshape(P*cap,D).T   # (1, P*cap)
slot     = ArgMax(logits)            # scalar in [0, P*cap)
token    = Vmap.reshape(P*cap)[slot] # original vocab id
```

`fused.py` (the Triton `block_sparse_argmax_atomic` kernel) is only a fused implementation of the gather→matmul→argmax→map steps with an int64-packed atomic-max; the math is identical. **Do not port the kernel for v1.**

**Cosine vs dot:** the reference normalizes centroids (`centroids / ‖centroids‖` along `D`) but **not** `h`. Replicate exactly — stage-1 selection is over `⟨h, unit_centroid_k⟩`. Normalizing `h` too would change nothing for argmax but do **not** normalize the centroids differently than the reference.

---

## 3. Offline assets (produced once per model)

These come from `clustering_cache.safetensors` in the reference (`centroids` `(D,K)`, `cluster_assignments` `(V,)`). The surgery consumes three derived tensors. **Fix these layouts — off-by-one or wrong transpose here silently corrupts outputs and is the hardest bug to find.**

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `Cnorm` | `(D, K)` | model dtype | Unit-normalized centroids as **columns**, so `MatMul(h, Cnorm) -> (1,K)`. (Reference stores `(K,D)` for `nn.Linear`; transpose it.) |
| `Wperm` | `(K, cap, D)` | see §7 | Original head rows reordered so block `k` holds the `cap` tokens of cluster `k`. |
| `Vmap` | `(K, cap)` | int64 | Original vocab id for each `(cluster, slot)`. `Wperm[k,s]` is the head row for token `Vmap[k,s]`. |

**Building them (Phase 1):**
- Run **balanced** k-means over the dense head rows (`W_vocab`, shape `(V,D)`) to get `K` clusters of exactly `cap` rows each. Easiest path: reuse the reference's published HuggingFace assets for a supported model, or implement balanced k-means (e.g. constrained assignment / auction). The balance invariant `K·cap == V` is **enforced** by the reference and required here.
- `Cnorm[:, k]` = normalized centroid of cluster `k`. `Wperm[k]` = the `cap` head rows of cluster `k`. `Vmap[k]` = their original vocab ids.

**Padding (gotcha):** real `V` rarely factors as `K·cap`. Pad the vocab to `V' = K·cap` with dummy rows whose logits are forced very negative (zero weight row + large-negative bias, or a post-hoc mask), and map dummy slots to a sentinel id that is filtered/never selected. Document `V'` vs `V`. **For the Qwen3-0.6B bootstrap this does not arise** — `V=151936=128×1187` divides exactly for `cap ∈ {16,32,64,128}` (see §0.1).

---

## 4. Target graph shape after surgery

```
                       hidden_state h_last (1, D)   [last position only]
                                  │
              ┌───────────────────┴───────────────────┐
   (prefill / seq>1: untouched dense head)     (decode FlashHead subgraph)
              │                                         │
        dense logits (1,seq,V)              Stage1 MatMul(Cnorm) → TopK(P)
                                                 → Gather(Wperm) → MatMul → ArgMax
                                                 → map slot→vocab id (+ special tokens)
                                                         │
                                            token id  OR  scattered (1,V) logits
```

- **Decode-only:** FlashHead applies to the **single last-position** hidden state. During prefill (`seq>1`) keep the dense head. The reference itself falls back to dense for long inputs (`hidden_states.shape[0] > 10`). v1 may simply leave prefill alone and only route the decode step.
- **Last-token slice:** most LLM ONNX exports already compute logits for the last position only (a `Gather`/slice on the sequence axis before the head). Reuse that hidden-state tensor as the FlashHead input.

---

## 5. Two integration contracts (pick one — key decision)

**A. Logits-shaped (recommended for `onnxruntime-genai` / existing samplers).** Emit a `(1, V)` logits tensor: pre-fill with a large-negative constant, `ScatterND`/`ScatterElements` the `P·cap` computed logits at their vocab ids. The downstream sampler/argmax (which often lives **outside** the ONNX graph in `genai_config.json` + the genai C++ loop, along with the KV cache) is unchanged — drop-in. Cost: an `O(V)` fill + scatter + the sampler's existing `O(V)` argmax/softmax each step. This is **elementwise over `V`**, not a matmul over `V·D`, so the big saving survives; the `O(V)` tail is small.

**B. Token-out (only for custom decode loops).** Emit the next-token id directly from the graph (as in §2). Cleanest and cheapest, but it **bypasses the external sampler**, so it breaks `onnxruntime-genai`'s loop and only works if you own the generation loop. Sampling/temperature must then also move into the graph.

> Recommendation: ship **A** first for compatibility and A/B-ability; consider **B** only if you control the loop and want to shave the `O(V)` tail.

---

## 6. Graph-surgery procedure (the hard part — detailed)

**6.1 Locate the head node.** Find the node whose output's last dim is `V` (the logits). It is typically one of: `MatMul`, `Gemm`, or the contrib op `com.microsoft.MatMulNBits` (block-quantized). Names often contain `lm_head`, `embed_out`, `logits`. Watch for **tied embeddings** (head weight == input embedding matrix, possibly only present as `embed_tokens.weight`). *For the genai-built Qwen3-0.6B:* the node is `/lm_head/MatMul` with an int8 weight (per §0.2), and because the model is tied, the same weights feed the input-embedding `Gather`/`GatherBlockQuantized` — so do **not** delete that weight when removing the head MatMul.

**6.2 Get the dense head weight `(V, D)`.**
- **Preferred:** take the **original bf16 head weight from the HF checkpoint** (pre-quantization). For Qwen3-0.6B it lives as `model.embed_tokens.weight` (tied), shape `(151936, 1024)`. Far simpler and avoids dequant error; cluster on this.
- **If only the quantized graph is available:** dequantize. For `MatMulNBits`, unpack the int4/int8 nibbles, apply per-block `scales` and (optional) `zero_points` using the contrib-op's block formula, and reshape to `(V, D)`. Use ORT tooling where possible; verify by comparing a few dequantized rows against a reference matmul. (Bootstrap recommendation: use the bf16 checkpoint and ignore graph dequant entirely.)

**6.3 Find the hidden-state input.** Trace the head node's data input (the `(…, D)` activation). Reuse the existing last-position slice if present; otherwise add a `Gather` on the sequence axis to take the last position during decode.

**6.4 Build the FlashHead subgraph** (from §2). Add `Cnorm`, `Wperm`, `Vmap` as initializers. Op notes:
- `TopK` `k` must be a **constant initializer** (static `P`).
- Prefer computing stage-2 logits as `MatMul(rows (P·cap, D), hT (D,1)) -> (P·cap,1)` (transpose the tiny `h`, not the big weight block).
- Match the graph's dtype (fp16 vs fp32). Insert `Cast` nodes as needed. **Accumulate stage-2 in fp32** if possible (the reference kernel accumulates in fp32); fp16 accumulation on some EPs diverges from the reference argmax.
- Keep `P`, `cap`, `K`, `D` static everywhere.

**6.5 Special-token / EOS safety (critical).** If EOS (or other required control tokens) lands in a cluster that isn't probed, **generation may never terminate**. Mirror the reference's `special_token_ids` path: maintain a small fixed "always-scored" block of special-token weight rows and concatenate their logits into stage-2 before the argmax/scatter. Treat EOS as mandatory.

**6.6 Rewire outputs.** Per the chosen contract (§5): replace the head node's consumers with either the token-id output (B) or the scattered `(1,V)` logits feeding the existing sampler (A). Remove the now-dead dense head node on the decode path only if you are not keeping it for prefill.

**6.7 Serialize.** Large initializers (`Wperm` is a second copy of the head weights ≈ `V·D` elements) will exceed the 2 GB protobuf limit for big vocabs — use **ONNX external-data** format. Re-run `onnx.checker` and shape inference.

---

## 7. Quantization handling (specific to your already-quantized model)

The single biggest gotcha: **you cannot cheaply `Gather` rows out of a `MatMulNBits`-packed weight** — its block layout interleaves packed nibbles with per-block scales/zero-points, so row selection doesn't line up.

- **v1:** keep `Wperm` as a **separate** tensor; do **not** reuse the packed head weights. Start `Wperm` in **fp16** (correctness first). This means carrying a second copy of the head weights → larger model (the probe's flash model was bigger for exactly this reason).
- **v2 (Phase 4):** quantize `Wperm` with a **gather-friendly** scheme — **per-row symmetric int8** — so you `Gather` the int8 rows **and** their per-row scales with the same indices, then dequantize the `P·cap` gathered rows only. Avoid block-quant for `Wperm`. Centroids (`Cnorm`, stage 1) tolerate low precision well (the reference's "selective quantization"): int8 centroids are fine.
- Once v2 holds accuracy, **delete the original dense head** on the decode path to recover the size overhead (keep it for prefill or also route prefill through a dense-on-`Wperm` matmul). **Note for Qwen3-0.6B:** because embeddings are tied, the original `(V,D)` weight must stay for the input embedding lookup regardless, so the `Wperm` copy is purely additive and cannot be reclaimed by deleting the head (see §0.1).

---

## 8. Phases, with exit criteria

**Phase 0 — Bootstrap & feasibility (see §0).** Build the int4/int8 Qwen3-0.6B model (§0.2), stand up the PPL + TTFT + TPS harness on **CPUExecutionProvider** (§0.4), record the baseline, **profile the `/lm_head/MatMul` decode-time share**, and run `flashhead_probe.py` at the real dims (`V=151936, D=1024, cap=16, K=9496`, sweep `P`). *Exit:* §0.5 — head share is material and the head-isolated probe is faster on CPU. (No GPU concerns since the whole effort is CPU; the batch-1 op-launch overhead that hurts FlashHead on GPU is not in play here.)

**Phase 1 — Asset generation.** Balanced k-means over `model.embed_tokens.weight` `(151936,1024)` → `Cnorm`, `Wperm`, `Vmap`; enforce `K·cap == V` (`cap=16, K=9496` — no padding). *Exit:* balance invariant holds; per-step top-1 agreement vs dense head ≥ target (e.g. ≥99%) on a hidden-state sample, at the chosen `P`.

**Phase 2 — Standalone subgraph with real tensors.** Reuse the probe builder; swap in real `Cnorm/Wperm/Vmap`. *Exit:* emitted token matches dense argmax at the Phase-1 agreement rate; fp32 accumulation parity confirmed.

**Phase 3 — Graph surgery into the real model.** §6 end-to-end with contract A. *Exit:* full model loads and runs on target EP; **full-sequence** exact-match / task metric within tolerance on held-out prompts (see §9); end-to-end decode latency improved; EOS terminates correctly.

**Phase 4 — Quantize `Wperm`.** Per-row int8 `Wperm` + int8 centroids; gather-with-scales. *Exit:* agreement/task metric maintained; model size reduced; original dense head removable on decode path.

**Phase 5 — Optional/future.** Distribution-preserving sampling (reference selects clusters proportional to centroid probabilities), custom fused ORT op for GPU, batch > 1, speculative/EAGLE, edge-EP tuning.

---

## 9. Validation & metrics

- **Per-step top-1 agreement** vs dense head (necessary, not sufficient).
- **Full-sequence exact match** under greedy: one divergent token cascades through context, so high per-step agreement can still yield low sequence match. This is the metric that matters.
- **PPL** — baseline dense quantized head is standard full-vocab PPL; FlashHead PPL is the probed-softmax variant + coverage (§0.4 caveat). Report both and the gap.
- **Latency:** TTFT and decode TPS via the genai harness, head-isolated probe latency, all with warmup + reps + median/IQR (§0.4).
- **Footprint:** model size, peak memory (note the `Wperm` second-copy in v1, unavoidable here due to tying).
- **EP:** CPU only for this effort (§0). Profile the head's decode-time share to bound the achievable model-level gain.

---

## 10. Caveats & gotchas (consolidated)

1. **MatMulNBits is not gatherable** → separate `Wperm`; v1 doubles head-weight footprint (§7).
2. **Balanced clusters required**; pad `V` to `K·cap` with forced-negative dummy tokens (§3) — **not needed for Qwen3-0.6B** (`V=151936` factors cleanly, §0.1).
3. **EOS/special tokens** must be always-scored or generation may never stop (§6.5).
4. **Edge EPs (QNN / NNAPI / CoreML)** have spotty support for `TopK` + **data-dependent `Gather`**; ops may silently fall back to CPU or fail to partition. Not in scope for the CPU bootstrap, but the highest-risk unknown **if/when** you later target edge — validate op coverage on the real EP before committing.
5. **GPU batch-1 launch overhead:** the un-fused op chain may not beat a single dense matmul on GPU; a fused custom op would be needed there. **Not relevant to this CPU-only effort** — plain ops realize the benefit on CPU (the probe shows ~5.7×).
6. **Amdahl:** model-level speedup is capped by the head's share of decode time. Measure that share first; the "up to 2×" assumes the head is ~half.
7. **Stage-1 cost:** `K` centroids are themselves scored every step (~`K/V` of the vocab). Large `K` erodes the benefit; tune `K` and `P` together.
8. **fp16 accumulation** in stage 2 can diverge from the reference's fp32-accumulated argmax → tie/near-tie flips. Accumulate in fp32 (§6.4).
9. **Argmax tie-breaking** differs across implementations; the reference's fused kernel packs `~vocab_id` to break ties toward the smallest id. Expect occasional benign mismatches on tied logits; don't count them as errors.
10. **Layout correctness:** `Wperm` row order must match `Vmap` exactly; a permutation mismatch corrupts outputs without crashing (§3).
11. **Logits-shaped contract** keeps an `O(V)` scatter + sampler tail each step — small vs `V·D`, but not zero (§5).
12. **`onnxruntime-genai` loop/KV-cache/sampler live outside the ONNX graph** (in `genai_config.json` + C++). Contract A is compatible; contract B requires owning the loop (§5).
13. **2 GB protobuf limit:** use external-data for `Wperm`/`Cnorm` at large vocab (§6.7).
14. **Recall vs `P`:** too-small `P` drops the true top-1 token; size `P` from the Phase-1 agreement curve, and re-check after quantization (Phase 4) since quant shifts scores.
15. **Tied embeddings:** the head weight may only exist as `embed_tokens.weight`; resolve before extraction (§6.1). For Qwen3-0.6B this is the norm, and it makes the `Wperm` copy unreclaimable (§0.1, §7).
16. **Builder int8-head + sharing on a tied model** (`k_quant_last`/`rtn_last` + auto `shared_embeddings`) is version-sensitive — verify the produced node dtypes rather than trusting the flags (§0.2).
17. **PPL is not apples-to-apples under FlashHead** — define the probed-softmax variant and report coverage; only baseline PPL is full-vocab (§0.4).

---

## 11. Suggested deliverable layout (for the implementing agent)

```
turbohead/
├── build/convert_qwen.sh           # §0.2 model-builder command (int4 + int8 head/embed)
├── eval/harness.py                 # PPL (bare-ORT logits) + TTFT/TPS (genai), warmup+reps
├── eval/head_share.py              # ORT-profile the /lm_head/MatMul decode-time share
├── assets/build_clusters.py        # balanced k-means on embed_tokens.weight -> Cnorm,Wperm,Vmap
├── surgery/extract_head.py         # locate head node / get (V,D) weight (prefer bf16 ckpt)
├── surgery/build_subgraph.py       # FlashHead subgraph (parametric P,cap,K,D; contract A/B)
├── surgery/splice.py               # rewire into the genai model; external-data save
├── quant/quantize_wperm.py         # Phase 4: per-row int8 Wperm + scales (gather-friendly)
├── eval/agreement.py               # per-step + full-sequence agreement + coverage
└── probe/flashhead_probe.py        # Phase 0 latency probe (provided)
```

## 12. Open decisions to resolve before coding

- Integration contract: **A (logits-shaped, genai-compatible)** vs **B (token-out)**. Recommend A for the bootstrap so the genai sampler/loop is untouched.
- `cap`/`K`/`P` operating point: start `cap=16, K=9496`; sweep `P` (e.g. 128 / 256 / 384 / 512) on the Phase-1 agreement curve.
- Whether to quantize `Wperm` in v1 or defer to Phase 4 (recommend defer; start fp16).
- Builder algo for the int8 head: `k_quant_last` (better accuracy) vs `rtn_last` (simpler) — decide after the §0.2 verification.

*(Resolved by this revision: target model = `Qwen/Qwen3-0.6B`; EP = CPU only; head-weight source = bf16 `embed_tokens.weight`; no vocab padding.)*
