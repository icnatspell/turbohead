# FlashHead — Current State & Next Steps

Status as of this revision. Target: `Qwen/Qwen3-0.6B`, INT4 body / INT8 head, **CPU only**.
Full spec in `PLAN.md`; reproduction in `../README.md`.

## Head-precision comparison

Full speed+quality matrix (dense head16/8/4 vs flash onnx/fused, across 1/2/4/8 threads, greedy +
sampling) lives in **`docs/RESULTS.md`**. Headline (Qwen3-0.6B, int4 body): fused contract-H is the
fastest path at 2.40×@1t vs fp32 head, beating every dense quant head; head8 g128 is the dense
quality sweet spot (98.5% agree, fp32-equal PPL). Reproduce: `turbohead-bench` + `turbohead-head-quality`.

## What works

End-to-end pipeline is implemented, reproducible, and correct:

Package layout: `surgery/` = apply the method (offline), `inference/` = run it (deploy), `eval/` = dev gates. Deps split to match (`turbohead` runtime-only; `turbohead[surgery,eval]` extras).

1. `surgery/convert_baseline.sh` — int4/int8 baseline (verified: body int4 `MatMulNBits`, head `/lm_head/MatMul_Q8` int8, tied embed via `GatherBlockQuantized`).
2. `surgery/extract_head.py` — bf16 head weight (tied = `embed_tokens.weight`).
3. `surgery/build_clusters.py` — **balanced** k-means (`cap=16`, `K=9496`, `K·cap=V` exact, no padding) via constrained Lloyd. Outputs `Cnorm (D,K)`, `Wperm (K,cap,D)`, `Vmap (K,cap)`.
4. `surgery/build_subgraph.py` — FlashHead subgraph, **contract A** (logits-shaped `(1,V)`, EOS always-scored). Stage-1 precision `fp16|int8|int4` (`MatMulNBits`, flexible `block_size`); stage-2 fp32 (gathered `Wperm`). `make_flash_nodes(stage1=…)` is the reusable splice fn + `quantize_stage1()`; `__main__` sweeps fp16/int8/int4 against the dense gate.
5. `surgery/splice.py` — last-position hidden → flash → `(1,1,V)`; dense head node removed (weight kept for tied embed); int4 stage-1 default; external-data save. `splice.py [P] [stage1] [block_size]`.
6. `eval/agreement.py` (quality) + `inference/decode_loop.py` (raw-ORT deploy loop: `Decoder` class, speed `--reps`, `--profile`, contract A/B auto-detect). genai dropped from the runtime — it's only the offline builder now.

## Measured results (P=256, single-thread CPU)

| Metric | Result |
|---|---|
| Per-token top-1 agreement vs dense (2000 WikiText-2 positions) | **97.6%** |
| Standalone subgraph vs dense argmax | **100%** (all-fp16; the int8 `Wperm` variant was 98.8%) |
| Model size | 717 MB (baseline 400 MB; +317 MB = additive fp16 `Wperm`, unreclaimable due to tied embed) |
| **Decode speed (median, raw-ORT loop)** | **1.21× @1 thread / 1.19× @4 threads** with int4-stage-1 (was 1.13× all-fp16) |
| Greedy full-sequence match vs baseline | 100% on deterministic prompts, ~2% on open-ended (the §9 cascade; both outputs valid) |

> **int4 stage-1 (current default).** Stage-1 centroid scoring (`fh_sims_mm`, the dominant head gemv at M=1) is now int4 `MatMulNBits` (W4A16, `accuracy_level=4`) — the paper's stage-1 trick. Isolated gemv 1.47ms→0.16ms (9×); end-to-end 1.13×→1.21× (1t), 100% standalone agreement preserved. Past int4 failures used `MatMulInteger`/manual dequant; `MatMulNBits` fuses dequant into the gemv (same op as the int4 body). Set via `splice.py 256 int4` (default); `int4` vs `fp16` selectable in `make_flash_nodes`.

> The all-fp16 result above supersedes an earlier int8-`Wperm` (Phase 4) build that measured ~1.0× (no speedup). Profiling showed the int8 path spent **~5ms/step on dequant** (`Cast`+`Mul` exploding the gathered int8 rows to fp32) — more than the matmuls themselves. Storing `Wperm` fp16 and doing stage-2 in fp16 removes that entirely: **faster *and* more accurate**, at the cost of disk size (569→717 MB). Phase-4 int8 `Wperm` is therefore a net negative on CPU/M=1 and was reverted.

## The headline finding (be honest about this)

**With the all-fp16 subgraph, FlashHead gives a real but modest ~1.10× decode speedup** at 97.6%/token agreement. The path to get here mattered:

- The baseline head is already **int8** (~155 MB, ~25% of decode time), not fp32. The reference's "up to 2×" and our early probe's 5.7× were vs an **fp32** dense head. Against int8 the Amdahl ceiling is only **~1.33×** — that is the hard cap on any head-only optimization here.
- Decode is **M=1 (gemv)**, memory-bound. Two dead ends, both measured: (a) **int8 `MatMulInteger`** both stages was *slower* — ORT's CPU int8 gemv kernel loses to its tuned fp16 gemv at M=1; (b) **int8 `Wperm` + dequant** (Phase 4) spent ~5ms/step on `Cast`+`Mul` materializing int8→fp32, more than the matmuls. **All-fp16 won both** (no dequant, no bad kernel): ~1.10×, 100% standalone correctness.
- Where the ~1.10× leaves money on the table → fusion (below). We are at ~1.10× of a ~1.33× ceiling.

## Where to pick up next time (priority order)

1. **Fused custom ORT op — the remaining headroom (~1.10× → toward ~1.33×).** *Assessed:* a meaningful version is a **C++ custom op** (Python/PyOp would be slower — pointless). Effort ~half-day standalone: write the kernel (gather fp16 rows → dot with `h` → top-1/scatter in one pass, no intermediate `(P·cap,D)` materialization, no per-op launch), build a shared lib against the ORT C API, register via `SessionOptions.RegisterCustomOpsLibrary`, and replace the `fh_*` nodes with the single op. **The real risk is genai integration** — its C++ loop must accept a custom-op library; if not, drive ORT directly with an own decode loop (which also enables contract B). De-risk first by checking genai custom-op support before writing the kernel. Profiling already shows the eliminable cost: stage-1 fp16 matmul ~1.8ms + stage-2 ~2ms + gather 0.6ms + the `O(V)` scatter — a fused pass should approach the ~1.33× ceiling.
2. **Try a model where the head is a larger share / fp32** — FlashHead's value scales with head share. Bigger vocab:hidden ratio, fewer layers, or an fp16/fp32 head → larger ceiling than 1.33×. **Re-run a head-share profile first** (was ~25% here; the Phase-0 `head_share.py` profiler was removed — recreate it: ORT `enable_profiling` on one decode step, sum `/lm_head*` node time / total).
3. **`cap=32` (K=4748)** — halves stage-1 `Cnorm` bandwidth (the 19MB/step fp16 read) and centroid count, at 2× stage-2 work. Cheap to try in `build_clusters.py` (change `CAP`); re-cluster + re-splice + re-bench.
4. **Probed-softmax sampling (temp≠0) — DONE (in the decode loop).** `decode_loop.py` now branches on temperature: `--temperature 0` = greedy argmax; `>0` = softmax/multinomial over only the scored candidate indices (`row > -1e8`), never the full vocab. Measured decode tail: greedy argmax over V = 0.013ms; **sampling over full V = 2.17ms**; over P·cap≈4096 candidates = 0.056ms. The dense baseline *must* softmax the full vocab every sampling step (2.17ms) — FlashHead skips it, so the advantage **grows in the sampling regime: ~1.21× greedy → ~1.35× sampling**. No graph change (the `(1,V)` output is `-1e9` outside candidates, so full-V softmax == candidate softmax). Verified: greedy unchanged/deterministic, sampling coherent + seed-varied, not slower than greedy. Contract-B (token-out) is greedy-only and raises on temp>0. Greedy token-out saves only ~0.11ms (~0.5%) — fold into the fused op #1, don't build standalone. *Remaining:* top-p/top-k filtering if needed (currently temperature only).
5. **FlashHead PPL** (probed-softmax + coverage, §0.4) — rigorous quality gate vs baseline full-vocab PPL, if pursuing further.

## State to resume from

- Working tree is the **all-fp16** build; nothing committed (still on `master`, no commits). `git add -A` then commit if you want a checkpoint.
- Artifacts on disk (gitignored, reproduce via README): `artifacts/qwen3_0_6b_int4_cpu/` (baseline), `artifacts/qwen3_0_6b_flash/` (spliced, P=256, 717MB), `artifacts/clusters.npz`, `artifacts/head_W.npy`.
- To re-measure: `uv run python turbohead/inference/decode_loop.py artifacts/qwen3_0_6b_flash --reps 5` (median of reps — single runs are pure noise, the source of an earlier spurious "+12%").

## Notes / gotchas confirmed

- Tied embeddings make `Wperm` purely additive; untying gives no size/speed/accuracy benefit for inference-only surgery.
- Clustering **quality** (not P) was the accuracy bottleneck: constrained Lloyd (in `kmeans()`) lifted P=256 agreement 92.9% → 97.6% at zero inference cost. Raising P past 256 *lowers* speed without meaningfully helping open-ended seq-match.
- ~~int8 anything on CPU at M=1 lost to fp16~~ — **corrected**: that was true for `MatMulInteger`/manual-dequant, but `MatMulNBits` (W4A16/W8A16, fused dequant) *wins* big at M=1 (int4 stage-1 9× on the gemv). Use `MatMulNBits` for any int gemv, never `MatMulInteger`.
- HF model-builder forces auth — pass `hf_token=false` in `--extra_options` for anonymous download of public repos.

## TODO (from my human)
0. Use a custom ORT loop instead of relying on onnxruntime-genai, similar to how we'd deploy. This way, we can try out Contract B (token out instead of logits out) and shave off O(V). This is fine since we'd deploy it like this anyhow.
1. See if we can run everything in INT, tied to (2).
2. Look at the paper and code implementation to see if we're missing anything or are doing something in a suboptimal manner (e.g., the method uses low precision for Stage 1 (coarse centroid scoring)). There may be some tricks they mention in either the paper or code that helps them get good acceleration. E.g., like what they do for the triton kernel.
3. Profile the bottlenecks in our case and think of how to overcome them.
4. Test across different # of CPU cores.
5. Check if we're using the DEFAULT_CLUSTER_RATIO:
    * Default ratio: number of clusters = vocab_size / DEFAULT_CLUSTER_RATIO (16 by default).
6. Verify if we can preserve a lot more task performance if we keep the head in FP16 and then can get the same speed back again with FlashHead, effectively making FlashHead an indirect method to recover PPL.
7. Cross-reference results with https://huggingface.co/spaces/embedl/Edge-Inference-Benchmarks. It seems like we may need to try it on different models that we care about.
    * They only show 1.15x speed-up for Qwen3-0.6B. This becomes 1.33x for Qwen3-1.7B. 1.35x for Gemma3-270M. 1.38x for Gemma3-1B. 1.25x for Qwen3.5-0.8B. 1.4x for Llama-3.2-1B.