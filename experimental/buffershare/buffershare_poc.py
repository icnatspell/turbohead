"""De-risk buffer-shared KV under raw ORT before wiring it into decode_loop.py.

GQA here takes seqlens_k/total_seq_len from the attention_mask subgraph and does RoPE internally
(do_rotary=1), so the only per-step growing input is the mask. Pre-allocate one [1,kv,MAX,hs] KV
buffer per layer, bind past_in and present_out to the SAME OrtValue (in-place write), grow the mask.

Checks: (1) tokens identical to the numpy round-trip path, (2) per-step slope vs S.
"""
import sys, time
import numpy as np
import onnxruntime as ort

BASE = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qwen3_0_6b/baseline"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 384
MAX = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS + 32

so = ort.SessionOptions(); so.intra_op_num_threads = 1
import os
lib = f"{BASE}/libturbohead.so"
if os.path.exists(lib): so.register_custom_ops_library(lib)
sess = ort.InferenceSession(f"{BASE}/model.onnx", so, providers=["CPUExecutionProvider"])
ins = {i.name: i for i in sess.get_inputs()}
outs = [o.name for o in sess.get_outputs()]
past_names = [n for n in ins if n.startswith("past")]
def present_of(n):
    pre = "past_key_values" if n.startswith("past_key_values") else "past"
    return "present" + n[len(pre):]
# shape of a KV buffer from the past input decl: [batch, kv_heads, seq, head]
sh = ins[past_names[0]].shape
kv_heads, head = sh[1], sh[3]
logits_name = "logits" if "logits" in outs else "cand_logits"
print(f"{BASE} | {len(past_names)} KV inputs | kv_heads={kv_heads} head={head} | MAX={MAX}")

PROMPT = [1, 2, 3, 4]


def greedy_pick(od):
    if "logits" in od:
        r = od["logits"].reshape(-1, od["logits"].shape[-1])[-1]
        return int(r.argmax())
    return int(od["cand_ids"].reshape(-1)[od["cand_logits"].reshape(-1).argmax()])


def buffered():
    # one persistent OrtValue per KV tensor, bound to BOTH past-in and present-out
    bufs = {n: ort.OrtValue.ortvalue_from_numpy(
        np.zeros((1, kv_heads, MAX, head), np.float32)) for n in past_names}
    io = sess.io_binding()
    bind_order = []                                        # get_outputs() follows binding order
    for n in past_names:
        io.bind_ortvalue_input(n, bufs[n])
        io.bind_ortvalue_output(present_of(n), bufs[n])   # same buffer -> in-place, persists
        bind_order.append(present_of(n))
    head_outs = [o for o in outs if not o.startswith("present")]

    def run(tokens, total):
        io.bind_cpu_input("input_ids", np.array([tokens], np.int64))
        m = np.zeros((1, MAX), np.int64); m[0, :total] = 1   # width=MAX (buffer stride), ones=valid
        io.bind_cpu_input("attention_mask", m)
        for o in head_outs:
            io.bind_output(o, "cpu")                       # dynamic shape -> rebind each step
        sess.run_with_iobinding(io)
        got = io.get_outputs()
        order = bind_order + head_outs
        return {o: got[order.index(o)].numpy() for o in head_outs}

    ids = list(PROMPT)
    od = run(ids, len(ids))
    nxt = greedy_pick(od); ids.append(nxt)
    rows, gen = [], [nxt]
    for _ in range(STEPS):
        S = len(ids)
        t = time.perf_counter()
        od = run([nxt], len(ids))
        rows.append((S, (time.perf_counter() - t) * 1000))
        nxt = greedy_pick(od); ids.append(nxt); gen.append(nxt)
    return rows, gen


def numpy_roundtrip():
    seed = {n: np.zeros([1, kv_heads, 0, head], np.float32) for n in past_names}
    def run(tokens, mask, past):
        feeds = {"input_ids": np.array([tokens], np.int64), "attention_mask": mask, **past}
        od = dict(zip(outs, sess.run(None, feeds)))
        return od, {n: od[present_of(n)] for n in past_names}
    ids = list(PROMPT)
    od, past = run(ids, np.ones((1, len(ids)), np.int64), seed)
    nxt = greedy_pick(od); ids.append(nxt); gen = [nxt]
    rows = []
    for _ in range(STEPS):
        S = len(ids)
        t = time.perf_counter()
        od, past = run([nxt], np.ones((1, len(ids)), np.int64), past)
        rows.append((S, (time.perf_counter() - t) * 1000))
        nxt = greedy_pick(od); ids.append(nxt); gen.append(nxt)
    return gen, rows


def slope(rows, warm=8):
    S = np.array([r[0] for r in rows], float); ms = np.array([r[1] for r in rows], float)
    b, a = np.polyfit(S[warm:], ms[warm:], 1)
    return a, b, ms


rows_buf, gen_buf = buffered()
gen_ref, rows_np = numpy_roundtrip()
match = sum(a == b for a, b in zip(gen_buf, gen_ref))
print(f"\ntoken match buffered vs numpy: {match}/{len(gen_ref)} "
      f"{'IDENTICAL' if gen_buf == gen_ref else 'DIVERGED at '+str(next(i for i,(a,b) in enumerate(zip(gen_buf,gen_ref)) if a!=b))}\n")
for label, rows in (("numpy round-trip", rows_np), ("buffer-shared ", rows_buf)):
    a, b, ms = slope(rows)
    print(f"{label}  floor {a:5.1f}ms  slope {b*1000:5.1f}ms/1000tok  "
          f"step@S{int(rows[-1][0])} {ms[-1]:5.1f}ms  growth {ms[-1]/ms[8]:.2f}x")
