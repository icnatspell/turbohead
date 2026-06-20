"""Measure how decode per-step latency grows with context length S, and split it into
the GQA attention-read term vs the KV-cache copy/framework term.

Method:
- drive the real fused model via Decoder._step, feed past back each step (the deployed path),
- record (S, step_ms) for S up to --steps,
- linear-fit step_ms ~ a + b*S  -> b is ms added per extra token of context (the O(S) slope),
- from an ORT node profile at long S, read GQA kernel ms/step (the attention floor); the rest
  of the slope is KV copy + framework overhead, i.e. the fixable part.

Also prints the KV traffic in bytes/token so the bandwidth floor is explicit.
"""
import sys, time, json, glob
import numpy as np
from turbohead.inference.decode_loop import Decoder

model_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qwen3_0_6b/fused"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 512

dec = Decoder(model_dir, threads=1, profile=False)
rng = np.random.default_rng(0)

# KV byte math (fp32 cache as exported)
bytes_per_tok = 2 * dec.n_layers * dec.kv_heads * dec.head_size * 4
print(f"{model_dir} | {dec.n_layers}L kv_heads={dec.kv_heads} head_size={dec.head_size}")
print(f"KV cache: {bytes_per_tok/1024:.1f} KB/token (fp32)  ->  at S={STEPS}: "
      f"{bytes_per_tok*STEPS/1e6:.0f} MB moved in+out per step\n")

# prefill a few tokens, then decode one-at-a-time recording per-step latency vs S
ids = [1, 2, 3, 4]
past, od = dec._step(np.array([ids], np.int64), np.ones((1, len(ids)), np.int64),
                     dec._empty_past(), 0)
nxt = dec._select(od, 0.0, rng)
ids.append(nxt)

rows = []
for _ in range(STEPS):
    S = len(ids)                      # past length seen by this step
    t = time.perf_counter()
    past, od = dec._step(np.array([[nxt]], np.int64),
                         np.ones((1, len(ids)), np.int64), past, len(ids) - 1)
    dt = (time.perf_counter() - t) * 1000
    nxt = dec._select(od, 0.0, rng)
    ids.append(nxt)
    rows.append((S, dt))

S = np.array([r[0] for r in rows], float)
ms = np.array([r[1] for r in rows], float)
# drop first few (warmup), fit a + b*S
warm = 8
b, a = np.polyfit(S[warm:], ms[warm:], 1)
print(f"step_ms ~ {a:.2f} + {b*1000:.3f} ms per 1000 ctx tokens")
for q in (1, 64, 128, 256, 384, 511):
    if q < len(ms):
        print(f"  S~{int(S[q]):5d}  step={ms[q]:6.2f} ms")
print(f"  growth S={int(S[warm])}->{int(S[-1])}: {ms[-1]-ms[warm]:+.2f} ms "
      f"({ms[-1]/ms[warm]:.2f}x)\n")

# node profile at long S: GQA kernel ms/step (attention floor) vs total
dec2 = Decoder(model_dir, threads=1, profile=True)
ids = list(range(2, 2 + 256))                       # long prefill so decode steps run at high S
past, od = dec2._step(np.array([ids], np.int64), np.ones((1, len(ids)), np.int64),
                      dec2._empty_past(), 0)
nxt = dec2._select(od, 0.0, rng)
ids.append(nxt)
for _ in range(32):
    past, od = dec2._step(np.array([[nxt]], np.int64),
                          np.ones((1, len(ids)), np.int64), past, len(ids) - 1)
    nxt = dec2._select(od, 0.0, rng)
    ids.append(nxt)
pf = dec2.sess.end_profiling()
ev = json.load(open(pf))
nodes = [e for e in ev if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")]
runs = [e["dur"] for e in ev if e.get("name") == "model_run"]
nrun = max(len(runs), 1)
import collections
by = collections.defaultdict(float)
for e in nodes:
    by[e["args"].get("op_name", "?")] += e["dur"]
tot = sum(by.values())
print(f"profile @ S~256-288 ({nrun} runs):")
for op, us in sorted(by.items(), key=lambda x: -x[1])[:8]:
    print(f"  {op:22s}{us/nrun/1000:7.2f} ms/step{us/tot*100:6.0f}%")
