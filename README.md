# TurboHead — fast LM head via ONNX graph surgery (Qwen3-0.6B, CPU)

Replace a quantized model's dense vocab projection with an approximate clustering-based head
("FlashHead": balanced k-means + multiprobe retrieval), by graph surgery on the ONNX model.
Stage-1 centroid scoring runs in int4/int8 `MatMulNBits`; stage-2 refines in fp32.

**1.21× decode @1 thread / 1.19× @4 threads** vs the int4 baseline, 100% top-1 agreement.
Full spec in `docs/PLAN.md`; findings + next steps in `docs/NEXT_STEPS.md`.

There are two separate workflows. **Applying** TurboHead to a model is a one-time offline
surgery. **Running inference** needs only a tiny ORT decode loop.

---

## A. Apply TurboHead to your model (offline, one-time)  →  `turbohead/surgery/`

Turns a base model into a spliced `…_flash/` ONNX model dir. Needs the surgery extras:

```bash
uv sync                                # dev install pulls surgery + eval extras + console scripts
# or, to apply only:  pip install "turbohead[surgery]"
```

```bash
# 1. build the int4/int8 baseline ONNX model   -> artifacts/qwen3_0_6b_int4_cpu/
bash turbohead/surgery/convert_baseline.sh

# 2. dump the bf16 head weight                 -> artifacts/head_W.npy
uv run turbohead-extract-head

# 3. balanced k-means clustering assets        -> artifacts/clusters.npz
uv run turbohead-build-clusters                # default cap=16 (cluster ratio); or --cap 32 / --clusters 4748

# 4. splice TurboHead into the model           -> artifacts/qwen3_0_6b_flash/
uv run turbohead-splice -P 256 --stage1 int4 --block-size 128
```

- **Cluster count** is the cluster ratio: `--cap N` = tokens per cluster (FlashHead's
  `DEFAULT_CLUSTER_RATIO=16`), giving `K = V/cap` clusters; or set `K` directly with `--clusters`.
  `cap` must divide `V` (151936 = 2⁷·1187 → cap ∈ {1,2,4,8,16,32,64,128}; default 16 → K=9496, no padding).
  Downstream (splice, agreement, decode) reads `cap`/`K` from the `.npz` — only this step needs the knob.
- `-P` (probes) and the **stage-1 precision** (`fp16` | `int8` | `int4`) + quant `--block-size` are
  chosen at splice time. int4 is the default and fastest; fp16 is the exact reference.
- Standalone check (subgraph argmax vs dense, sweeps fp16/int8/int4):
  `uv run turbohead-build-subgraph -P 256`.

## B. Run inference  →  `turbohead/inference/`

The deploy path. Self-contained: **onnxruntime + numpy + a tokenizer, no genai, no torch.**
`pip install turbohead` is enough.

```bash
uv run turbohead-decode artifacts/qwen3_0_6b_flash --reps 5
```

```python
from turbohead.inference.decode_loop import Decoder
dec = Decoder("artifacts/qwen3_0_6b_flash", threads=1)   # dims/contract/tokenizer auto-detected
ids = dec.tok("Once upon a time,")["input_ids"]
out, tok_s = dec.generate(ids, max_new=64)
print(dec.tok.decode(out))
```

The loop auto-detects the head contract (A = logits-out, B = token-out) and manages the KV
cache itself. Flags: `--threads N`, `--reps R` (median tok/s), `--max-new M`, `--profile`
(per-op decode-step breakdown), `--prompt STR`, `--temperature T` / `--seed N`.

`--temperature 0` is greedy argmax; `>0` samples via **probed-softmax** — softmax/multinomial
over only the scored candidate set, skipping the ~2 ms full-vocab softmax the dense head pays
each step, so FlashHead's speedup is *larger* when sampling (~1.35×) than greedy (~1.21×).

## C. Quality + speed gates (dev)  →  `turbohead/eval/`

```bash
uv run turbohead-agreement --npz artifacts/clusters.npz   # top-1 agreement vs dense (WikiText-2)
```
