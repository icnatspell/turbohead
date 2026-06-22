"""A/B: same baseline graph, two runtimes. Isolates the avoidable KV-copy term.

- raw ORT (our deploy path): numpy round-trip of the full KV every step -> append-mode GQA.
- onnxruntime-genai runtime: buffer-shared static KV, writes new k/v in place, no realloc.

Same ops, same weights, same 1 thread. Difference in the per-step-vs-S slope = the copy overhead
we can recover by switching the body to a buffer-shared export.
"""
import sys, time
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qwen3_0_6b/baseline"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 384


def fit(rows, warm=8):
    S = np.array([r[0] for r in rows], float)
    ms = np.array([r[1] for r in rows], float)
    b, a = np.polyfit(S[warm:], ms[warm:], 1)
    return a, b, ms


def raw_ort():
    from turbohead.inference.decode_loop import Decoder
    dec = Decoder(BASE, threads=1, profile=False)
    rng = np.random.default_rng(0)
    ids = [1, 2, 3, 4]
    past, od = dec._step(np.array([ids], np.int64), np.ones((1, len(ids)), np.int64),
                         dec._empty_past(), 0)
    nxt = dec._select(od, 0.0, rng); ids.append(nxt)
    rows = []
    for _ in range(STEPS):
        S = len(ids)
        t = time.perf_counter()
        past, od = dec._step(np.array([[nxt]], np.int64),
                             np.ones((1, len(ids)), np.int64), past, len(ids) - 1)
        rows.append((S, (time.perf_counter() - t) * 1000))
        nxt = dec._select(od, 0.0, rng); ids.append(nxt)
    return rows


def genai():
    import onnxruntime_genai as og
    model = og.Model(BASE)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=STEPS + 16)
    gen = og.Generator(model, params)
    gen.append_tokens([1, 2, 3, 4])
    gen.generate_next_token()                      # prefill + first token
    rows = []
    for _ in range(STEPS):
        S = len(gen.get_sequence(0))
        t = time.perf_counter()
        gen.generate_next_token()
        rows.append((int(S), (time.perf_counter() - t) * 1000))
        if gen.is_done():
            break
    return rows


print(f"baseline={BASE} | 1 thread | {STEPS} decode steps\n")
for name, fn in (("raw ORT (numpy KV round-trip)", raw_ort), ("genai (buffer-shared KV)", genai)):
    a, b, ms = fit(fn())
    print(f"{name}")
    print(f"  step_ms ~ {a:.2f} + {b*1000:.3f} ms / 1000 ctx tok   "
          f"(S~16:{ms[16]:.1f}ms  S~{len(ms)-1+16}:{ms[-1]:.1f}ms  {ms[-1]/ms[8]:.2f}x)\n")
