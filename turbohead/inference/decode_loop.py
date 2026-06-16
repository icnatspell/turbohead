"""TurboHead production decode path — raw onnxruntime, no onnxruntime-genai.

This is the ONLY file you need to run inference on a turbohead-spliced model. It is
self-contained (onnxruntime + numpy + a tokenizer); the surgery/ package is not imported.

Greedy decode over a Qwen3 ONNX decoder via InferenceSession + manual KV cache.
genai is only the offline baseline builder (see surgery/convert_baseline.sh).

Two head contracts, auto-detected from the graph's outputs:
  A (logits-out): graph emits `logits` (1,1,V); argmax taken here.   [current spliced graph]
  B (token-out):  graph emits the next-token id directly; V never materialized.

Doubles as the profiler (--profile): per-op decode-step breakdown via ORT profiling.

CLI:
  uv run python turbohead/inference/decode_loop.py <model_dir> [opts]
    --threads N    intra-op threads (default 1)
    --max-new M    tokens to generate (default 64)
    --reps R       benchmark: median tok/s over R timed runs (default 1 = single run)
    --profile      dump + summarize an ORT profile of one run
    --prompt STR   prompt text
    --temperature  0 = greedy argmax; >0 = sample over the probed candidate set (skips the
                   ~2ms full-vocab softmax the dense head pays — see docs/NEXT_STEPS.md #4)
    --seed N       RNG seed for sampling
"""
import os
import time
import json
import glob
import argparse
import collections
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from loguru import logger

DEFAULT_PROMPT = "Once upon a time, in a small village,"

# Flash/head-path node-name fragments, for the profiler's head rollup (best-effort).
HEAD_NODE_HINTS = ("lm_head", "fh", "flash", "scatter", "gather", "argmax", "topk")


class Decoder:
    """Loads a Qwen3 ONNX decoder and runs greedy decode with a manual KV cache.

    Dims (layers, KV heads, head size), head contract, tokenizer and EOS are all
    discovered from the model dir — nothing model-specific is hardcoded.

    KV cache flows present->past as numpy. ponytail: IOBinding zero-copy KV was tried and
    is a net loss here — contract A must pull `logits` to numpy for argmax anyway, and
    reusing ORT-allocated output buffers as next-step inputs is a use-after-free (ORT
    recycles the arena -> segfault). IOBinding pays off only under contract B (token-out,
    tiny outputs, everything device-side); revisit it when the token-out graph exists.
    """

    def __init__(self, model_dir, threads=1, profile=False):
        self.dir = model_dir.rstrip("/")
        # match the fused kernel's OpenMP threads to ORT intra-op (read by libgomp at
        # first parallel region; set before the lib runs). Per-process, fine for serving.
        os.environ["OMP_NUM_THREADS"] = str(threads)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        if profile:
            so.enable_profiling = True
            so.profile_file_prefix = f"{self.dir}/ortprof"
        lib = f"{self.dir}/libturbohead.so"  # fused model ships its custom-op kernel here
        if os.path.exists(lib):
            so.register_custom_ops_library(lib)
        self.sess = ort.InferenceSession(f"{self.dir}/model.onnx", so,
                                         providers=["CPUExecutionProvider"])
        self.out_names = [o.name for o in self.sess.get_outputs()]

        # Tokenizer + EOS ship with the model (splice copies them) — no hardcoded identity.
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        eos = json.load(open(f"{self.dir}/genai_config.json"))["model"].get(
            "eos_token_id", self.tok.eos_token_id)
        self.eos = set(eos) if isinstance(eos, list) else {eos}

        # KV cache geometry from the past_key_values.*.key inputs.
        keys = [i for i in self.sess.get_inputs() if i.name.startswith("past_key_values.")]
        self.n_layers = sum(1 for k in keys if k.name.endswith(".key"))
        kshape = next(i.shape for i in self.sess.get_inputs()
                      if i.name == "past_key_values.0.key")
        self.kv_heads, self.head_size = int(kshape[1]), int(kshape[3])
        self.kv_names = [f"past_key_values.{i}.{kv}"
                         for i in range(self.n_layers) for kv in ("key", "value")]
        self.present_names = [f"present.{i}.{kv}"
                              for i in range(self.n_layers) for kv in ("key", "value")]

        # Contract: A emits full `logits` (1,V); H emits the candidate shortlist
        # (cand_logits, cand_ids) from the fused op; B emits the token id directly.
        self.contract = ("A" if "logits" in self.out_names
                         else "H" if "cand_logits" in self.out_names else "B")
        self.token_out = ("logits" if self.contract == "A"
                          else None if self.contract == "H"
                          else next(n for n in self.out_names if not n.startswith("present.")))

    def _empty_past(self):
        z = np.zeros((1, self.kv_heads, 0, self.head_size), np.float32)
        return dict.fromkeys(self.kv_names, z)

    def _step(self, input_ids, attn, past):
        od = dict(zip(self.out_names,
                      self.sess.run(None, {"input_ids": input_ids,
                                           "attention_mask": attn, **past})))
        past = {p: od[pr] for p, pr in zip(self.kv_names, self.present_names)}
        return past, od

    @staticmethod
    def _pick(logits, ids, temperature, rng):
        """Greedy (argmax) or temperature sampling over a candidate (logits, ids) shortlist."""
        if not temperature:
            return int(ids[logits.argmax()])
        x = logits.astype(np.float32) / temperature
        x -= x.max()
        p = np.exp(x)
        p /= p.sum()
        return int(ids[rng.choice(ids.size, p=p)])

    def _select(self, od, temperature, rng):
        """Pick the next token. H: argmax/sample over the fused op's shortlist. A: same,
        but the candidate set is the (1,V) entries above the -1e9 fill (skips full-vocab
        softmax). B: token-out, greedy only."""
        if self.contract == "H":
            return self._pick(od["cand_logits"].reshape(-1), od["cand_ids"].reshape(-1),
                              temperature, rng)
        if self.contract == "B":
            if temperature:
                raise ValueError("temperature>0 needs logits; this graph is token-out (contract B)")
            return int(np.asarray(od[self.token_out]).flat[-1])
        row = np.asarray(od[self.token_out])                 # contract A: (1,1,V) logits
        row = row.reshape(-1, row.shape[-1])[-1]              # (V,) last position
        cand = np.flatnonzero(row > -1e8)                    # scored candidates (skip -1e9 fill)
        return self._pick(row[cand], cand, temperature, rng)

    def generate(self, prompt_ids, max_new, temperature=0.0, seed=None):
        """Returns (generated_ids, decode_tok_per_s). Decode timer excludes prefill.
        temperature=0 = greedy argmax; >0 = sample over the probed candidate set."""
        rng = np.random.default_rng(seed)
        ids = list(prompt_ids)
        n0 = len(ids)
        past, od = self._step(np.array([ids], np.int64),
                               np.ones((1, len(ids)), np.int64), self._empty_past())
        nxt = self._select(od, temperature, rng)
        ids.append(nxt)
        t1 = time.perf_counter()
        gen = 1
        while nxt not in self.eos and gen < max_new:
            past, od = self._step(np.array([[nxt]], np.int64),
                                   np.ones((1, len(ids)), np.int64), past)
            nxt = self._select(od, temperature, rng)
            ids.append(nxt)
            gen += 1
        return ids[n0:], gen / (time.perf_counter() - t1)


def summarize_profile(model_dir, n_warmup=2):
    """Per-op decode-step breakdown from the latest ORT profile json.

    Node kernel times are summed over the whole run and divided by #model_run events
    for an avg/step (prefill is amortized in — flagged below). The model_run medians
    give the clean per-step and prefill latencies.
    """
    f = sorted(glob.glob(f"{model_dir.rstrip('/')}/ortprof*.json"))[-1]
    events = json.load(open(f))
    runs = [e["dur"] for e in events if e.get("name") == "model_run"]
    nodes = [e for e in events if e.get("cat") == "Node"
             and e.get("name", "").endswith("_kernel_time")]

    prefill = runs[0] / 1000 if runs else 0.0
    decode = runs[1 + n_warmup:] or runs[1:] or runs
    by_op = collections.defaultdict(lambda: [0.0, 0])
    head_us = 0.0
    for e in nodes:
        op = e["args"].get("op_name", "?")
        by_op[op][0] += e["dur"]
        by_op[op][1] += 1
        if any(h in e["name"].lower() for h in HEAD_NODE_HINTS):
            head_us += e["dur"]
    nruns = max(len(runs), 1)
    total_us = sum(v[0] for v in by_op.values())

    logger.info(f"profile: {f}")
    logger.info(f"  prefill model_run={prefill:.2f} ms | "
                f"decode model_run median={np.median(decode) / 1000:.2f} ms "
                f"(n={len(decode)})")
    logger.info(f"  head-path nodes ~{head_us / nruns / 1000:.2f} ms/step "
                f"({head_us / total_us * 100:.0f}% of node time) — best-effort by name")
    logger.info(f"  {'op_type':24s}{'ms/step':>9}{'% node':>8}{'calls':>8}")
    for op, (us, n) in sorted(by_op.items(), key=lambda x: -x[1][0])[:12]:
        logger.info(f"  {op:24s}{us / nruns / 1000:9.2f}{us / total_us * 100:7.0f}%{n:8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--temperature", type=float, default=0.0)  # 0 = greedy; >0 = probed-softmax sample
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    dec = Decoder(a.model_dir, a.threads, a.profile)
    ids = dec.tok(a.prompt)["input_ids"]
    logger.info(f"{a.model_dir} | contract {dec.contract} | "
                f"{dec.n_layers}L kv_heads={dec.kv_heads} head_size={dec.head_size} | "
                f"threads={a.threads} | temp={a.temperature}")

    rates, out = [], None
    for r in range(a.reps + (a.reps > 1)):        # one warmup when benchmarking
        out, tps = dec.generate(ids, a.max_new, temperature=a.temperature, seed=a.seed)
        if a.reps == 1 or r > 0:
            rates.append(tps)
    import statistics
    logger.info(f"decode {statistics.median(rates):.1f} tok/s "
                f"(median of {len(rates)}) | {len(out)} toks")
    logger.info(f"  -> {dec.tok.decode(out)!r}")
    assert out and out[0] != ids[-1], "decode produced nothing sane"

    if a.profile:
        dec.sess.end_profiling()
        summarize_profile(a.model_dir)


if __name__ == "__main__":
    main()
