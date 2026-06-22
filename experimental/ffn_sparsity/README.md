# ffn_sparsity

**Status: PARKED — ~1.2x ceiling on the int4 body, not worth the custom kernel. Revisit if body→fp16.**

## Objective

Apply the FlashHead idea (cheap predictor + Gather + dense matmul) to the next bottleneck after the
head splice: the SwiGLU FFN, which is ~49% of the Qwen3-0.6B decode step. Predict the active
intermediate neurons, gather only those rows of `gate_proj`/`up_proj`, run the smaller matmul +
activation, gather the matching columns of `down_proj` (TEAL / SparseInfer training-free sparsity).

## The gate (`ffn_sparsity_poc.py`)

Measures the **oracle ceiling**: monkeypatches each MLP forward to compute the true intermediate
`act(gate(x))*up(x)`, keep only the top-k magnitude neurons, zero the rest, run `down_proj`. Reports
end-to-end top-1 agreement vs the full model over WikiText-2, so cross-layer drift compounds (FFN
sparsity approximates in *every* layer, unlike the head's one splice). A real deploy can't see true
magnitudes — it predicts the active set first — so this is an upper bound. `keep=100%`→100% agreement
is the built-in self-check. ~90s on Qwen3-0.6B.

```bash
uv run python experimental/ffn_sparsity/ffn_sparsity_poc.py [--model Qwen/Qwen3-0.6B]
```

## Result (2026-06-21, Qwen3-0.6B)

FFN ≈ **49% of the decode step** (profile: MatMulNBits 37ms/75%, FFN ~60% of body matmul by
param-bytes). Oracle sparsity vs top-1 agreement: 30%→97.3%, 40%→96.1%, 50%→93.7%, 60%→90.5%,
70%→84.9%. Amdahl at 50% sparsity / ideal 2x on FFN = **1.32x ceiling**; realistic ~1.1–1.2x
single-threaded, eroding with threads.

## Why parked

On an int4 body, int4 (4x byte saving) and row-gather sparsity don't stack with standard ONNX ops —
gathering arbitrary rows of block-quantized `MatMulNBits` weights needs a custom kernel (storing fp16
to make Gather work reads *more* bytes than the int4 dense matmul at 40–50% sparsity). Even with the
kernel, the two FFN halves are asymmetric: `gate`/`up` select **output rows** (N-major, gather-friendly)
but `down`'s active neurons are the **contraction axis** with int4 blocks misaligned → can't cheaply
skip, so the realistic win is ~1.5x not 2x. Add per-layer predictor cost, the real predictor missing
oracle agreement (forces lower sparsity to hold quality), compounding accuracy loss across 28 layers,
and thread-fragile gains. Multi-week kernel surface for ~1.2x that stacks weakly on the head's shipped
2.40x. Spark Transformer's 1.35–1.79x (16-core, real kernel) is the reference.

## When to revisit

- **Body moves to fp16** — then plain Gather+MatMul works (no kernel) and the 49% share makes it worth it.
- **Higher natural sparsity** (ReLU / dReLU-tuned models à la Turbo Sparse ~89%) — re-run the gate first.
