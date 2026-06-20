"""TurboHead production decode path — raw onnxruntime, no onnxruntime-genai.

This is the ONLY file you need to run inference on a turbohead-spliced model. It is
self-contained (onnxruntime + numpy + a tokenizer); the surgery/ package is not imported.

Greedy or temperature-sampling decode over any genai-style ONNX decoder via InferenceSession
+ manual KV cache. Dims, head contract, state layout, tokenizer and EOS are all discovered from
the model dir — standard, hybrid (conv/SSM + attention) and embeds-in models, no per-model config.
genai is only the offline baseline builder (see surgery/convert_baseline.sh).

Three head contracts, auto-detected from the graph's outputs:
  A (logits-out):   graph emits `logits` (1,1,V); argmax/sample here.        [onnx backend]
  H (shortlist-out): the fused op emits (cand_logits, cand_ids); V never materialized. [fused]
  B (token-out):    graph emits the next-token id directly; greedy only.

Doubles as the profiler (--profile): per-op decode-step breakdown via ORT profiling.

CLI:
  uv run python turbohead/inference/decode_loop.py <model_dir> [opts]
    --threads N    intra-op threads (default 1)
    --max-new M    tokens to generate (default 64)
    --reps R       benchmark: median tok/s over R timed runs (default 1 = single run)
    --profile      dump + summarize an ORT profile of one run
    --prompt STR   prompt text
    --temperature  0 = greedy argmax; >0 = sample over the probed candidate set (skips the
                   ~2ms full-vocab softmax the dense head pays — see docs/ORT_QUIRKS.md #6)
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

    KV cache: by default flows present->past as numpy (a full copy of the cache in and out
    every step -> O(S) traffic/step, the dominant cost at long context). `share_kv` instead
    pre-allocates one max-length OrtValue per growing KV tensor and binds past-in and
    present-out to it via IOBinding, so GQA writes the new k/v IN PLACE (seqlens come from a
    MAX-width attention_mask). Byte-identical output, ~halves the per-step-vs-length slope
    (logs/buffershare_poc.py). The earlier IOBinding dead-end was reusing ORT-*allocated*
    output buffers as next inputs (arena recycle -> segfault); a persistent *user*-allocated
    buffer bound to both sides is the supported pattern and sidesteps that. Auto-enabled for
    pure-KV transformers; hybrids with fixed conv/SSM state fall back to the numpy path.

    Shared-KV limit (see docs/IDEAS.md #6): the buffer is FIXED-length (`max_kv`, default =
    this request's prompt+max_new). It is also the memory ceiling (~224 KB/token on Qwen3-0.6B).
    A generation that would exceed it is clamped and STOPS EARLY -- we do not drop old tokens.
    We must not write past the buffer: GQA writes each k/v in place at offset=past_seq_len, so an
    over-long write lands outside the arena (OOB -> segfault/corruption), not a realloc. To go
    past the cap you need sliding-window eviction, which is NOT cheap here: GQA ties RoPE-position,
    write-offset and seqlens to one value, so a ring buffer is impossible -- you must left-shift
    the cache (a copy) AND re-rotate every surviving key into the compacted position frame (exact
    per-model RoPE; lossy, no longer byte-identical). ~40-80 lines; build only for real long-chat.
    """

    def __init__(self, model_dir, threads=1, profile=False, share_kv=True, max_kv=None):
        self.dir = model_dir.rstrip("/")
        self.max_kv = max_kv          # fixed shared-KV buffer length (None = size to each request)
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
        self.in_names = {i.name for i in self.sess.get_inputs()}

        # Tokenizer + EOS ship with the model (splice copies them) — no hardcoded identity.
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        eos = json.load(open(f"{self.dir}/genai_config.json"))["model"].get(
            "eos_token_id", self.tok.eos_token_id)
        self.eos = set(eos) if isinstance(eos, list) else {eos}

        # State inputs: KV cache and/or SSM conv/recurrent state. Discovered generically by name
        # so hybrid models work — attention layers interleaved with conv/recurrent layers, at
        # sparse indices (e.g. LFM2, Qwen3.5), not just uniform-KV transformers. Each `past*` input
        # is fed back each step from the matching `present*` output (same suffix), seeded with zeros
        # of its OWN declared shape: KV's seq dim is symbolic -> 0 (grows), recurrent/conv state is
        # all-concrete -> full size (updated in place). One path handles both.
        state = [i for i in self.sess.get_inputs() if i.name.startswith("past")]
        self.state_seed = {i.name: self._zero_state(i) for i in state}
        # present output for each past input: past_key_values.2.key -> present.2.key,
        # past_conv.0 -> present_conv.0 (strip the `past[_key_values]` prefix, prepend `present`).
        def present_of(n):
            pre = "past_key_values" if n.startswith("past_key_values") else "past"
            return "present" + n[len(pre):]
        self.present_for = {i.name: present_of(i.name) for i in state}
        missing = [p for p in self.present_for.values() if p not in self.out_names]
        assert state and not missing, f"no present output for {missing}"
        # geometry (profile log only): attention layers carry `.key`; SSM layers don't.
        keys = [i for i in state if i.name.endswith(".key")]
        self.n_layers = len(keys)
        self.kv_heads, self.head_size = (int(keys[0].shape[1]), int(keys[0].shape[3])) if keys else (0, 0)

        # Buffer-shared KV (see class docstring): only for pure-KV transformers — every state input
        # must have a symbolic (growing) seq dim, and the graph must take attention_mask (it drives
        # GQA's seqlens, so a MAX-width mask sets the in-place buffer stride). Hybrids with fixed-size
        # conv/SSM state, or graphs without attention_mask, keep the numpy feedback path.
        all_growing = all(any(not isinstance(d, int) for d in i.shape[1:]) for i in state)
        self.share_kv = share_kv and all_growing and "attention_mask" in self.in_names
        self.head_out_names = [o for o in self.out_names if not o.startswith("present")]

        # Token feed: most graphs take input_ids; some (Qwen3.5) split the embedding lookup out and
        # want inputs_embeds — we do the lookup in numpy from the tied embedding (= head_W) — plus a
        # position_ids (Qwen3.5's is 3-D, M-RoPE; for text all channels = the absolute position).
        extra = (self.in_names - {"input_ids", "inputs_embeds", "attention_mask", "position_ids"}
                 - set(self.state_seed))
        if extra or not ({"input_ids", "inputs_embeds"} & self.in_names):
            raise NotImplementedError(f"{self.dir}: unsupported graph inputs {sorted(extra)} "
                                      "(need input_ids or inputs_embeds, + attention_mask + state)")
        self.embeds_in = "inputs_embeds" in self.in_names
        self.wants_position_ids = "position_ids" in self.in_names
        self.pos_rank = next((len(i.shape) for i in self.sess.get_inputs()
                              if i.name == "position_ids"), 0)
        if self.embeds_in:  # tied-embedding lookup matrix: model dir, else build_all's artifacts/<slug>/
            ep = next((p for p in (f"{self.dir}/embed.npy", f"{self.dir}/../head_W.npy")
                       if os.path.exists(p)), None)
            if not ep:
                raise FileNotFoundError(f"{self.dir}: inputs_embeds graph needs embed.npy or ../head_W.npy")
            self.embed = np.load(ep, mmap_mode="r")  # [V,D] tied embed; fancy-index gathers token rows

        # Contract: A emits full `logits` (1,V); H emits the candidate shortlist
        # (cand_logits, cand_ids) from the fused op; B emits the token id directly.
        # `backend` is the human-facing label for which splice produced this model.
        self.contract = ("A" if "logits" in self.out_names
                         else "H" if "cand_logits" in self.out_names else "B")
        self.backend = {"A": "onnx", "H": "fused", "B": "fused"}[self.contract]
        self.token_out = ("logits" if self.contract == "A"
                          else None if self.contract == "H"
                          else next(n for n in self.out_names if not n.startswith("present.")))

    @staticmethod
    def _zero_state(inp):
        """Zero seed for a state input from its declared shape: dim 0 (batch) -> 1, any other
        symbolic dim -> 0 (KV seq dim, grows); concrete dims kept (recurrent/conv state, full size)."""
        dt = {"tensor(float)": np.float32, "tensor(float16)": np.float16}.get(inp.type, np.float32)
        shape = [d if isinstance(d, int) else (1 if i == 0 else 0) for i, d in enumerate(inp.shape)]
        return np.zeros(shape, dt)

    def _empty_past(self):
        return dict(self.state_seed)

    def _pos(self, start, k):
        """position_ids for the k tokens at absolute positions [start, start+k). Matches the graph's
        declared rank — 3 = Qwen3.5 M-RoPE [3,1,k] (text: all channels equal), else 2-D [1,k]."""
        p = np.arange(start, start + k, dtype=np.int64)
        return np.broadcast_to(p, (3, 1, k)).copy() if self.pos_rank == 3 else p.reshape(1, k)

    def _step(self, tokens, attn, past, start):
        # tokens: int64 (1,k). embeds-in graphs get inputs_embeds = tied-embed rows; else input_ids.
        x = self.embed[tokens].astype(np.float32, copy=False) if self.embeds_in else tokens
        feeds = {("inputs_embeds" if self.embeds_in else "input_ids"): x, "attention_mask": attn, **past}
        if self.wants_position_ids:
            feeds["position_ids"] = self._pos(start, tokens.shape[1])
        od = dict(zip(self.out_names, self.sess.run(None, feeds)))
        past = {name: od[pres] for name, pres in self.present_for.items()}
        return past, od

    def _setup_shared(self, max_len):
        """Pre-allocate one OrtValue per growing KV tensor (seq dim -> max_len) and bind past-in and
        present-out to the SAME buffer, so GQA updates the cache in place. io.get_outputs() follows
        binding order, so track it to find head outputs afterwards. max_len is also the mask width."""
        self._max_len = max_len
        self._bufs = {}
        order = []
        for i in self.sess.get_inputs():
            if i.name not in self.present_for:
                continue
            dt = {"tensor(float)": np.float32, "tensor(float16)": np.float16}.get(i.type, np.float32)
            shape = [1 if j == 0 else (d if isinstance(d, int) else max_len)
                     for j, d in enumerate(i.shape)]
            self._bufs[i.name] = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, dt))
        self._io = self.sess.io_binding()
        for name, buf in self._bufs.items():
            self._io.bind_ortvalue_input(name, buf)
            self._io.bind_ortvalue_output(self.present_for[name], buf)
            order.append(self.present_for[name])
        self._out_order = order + self.head_out_names

    def _step_shared(self, tokens, total, start):
        x = self.embed[tokens].astype(np.float32, copy=False) if self.embeds_in else tokens
        self._io.bind_cpu_input("inputs_embeds" if self.embeds_in else "input_ids", x)
        m = np.zeros((1, self._max_len), np.int64)   # width = buffer stride; ones = valid length
        m[0, :total] = 1
        self._io.bind_cpu_input("attention_mask", m)
        if self.wants_position_ids:
            self._io.bind_cpu_input("position_ids", self._pos(start, tokens.shape[1]))
        for o in self.head_out_names:
            self._io.bind_output(o, "cpu")           # head outputs are dynamic -> rebind each step
        self.sess.run_with_iobinding(self._io)
        got = self._io.get_outputs()
        return {o: got[self._out_order.index(o)].numpy() for o in self.head_out_names}

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
        if self.share_kv:
            need = n0 + max_new + 1
            cap = self.max_kv or need
            if n0 >= cap:
                raise ValueError(f"prompt ({n0}) exceeds shared-KV cap ({cap}); raise max_kv")
            # ponytail: at the cap we clamp and stop early, NOT drop-oldest. Sliding-window eviction
            # needs cache shift + per-model RoPE re-rotation (GQA ties pos==write-offset==seqlen);
            # ~40-80 lossy lines -- see class docstring / docs/IDEAS.md #6. Add for real long-chat.
            max_new = min(max_new, cap - n0 - 1)   # clamp so GQA never writes past the buffer
            self._setup_shared(cap)

        def step(toks, start, past):           # numpy path threads past in/out; shared path ignores it
            if self.share_kv:
                return None, self._step_shared(toks, len(ids), start)
            return self._step(toks, np.ones((1, len(ids)), np.int64), past, start)

        past = self._empty_past()
        past, od = step(np.array([ids], np.int64), 0, past)
        nxt = self._select(od, temperature, rng)
        ids.append(nxt)
        t1 = time.perf_counter()
        gen = 1
        while nxt not in self.eos and gen < max_new:
            past, od = step(np.array([[nxt]], np.int64), len(ids) - 1, past)
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
    ap.add_argument("--no-share-kv", action="store_true", help="disable in-place buffer-shared KV")
    ap.add_argument("--max-kv", type=int, default=None,
                    help="fixed shared-KV buffer length, e.g. 2048 (default: size to each request)")
    a = ap.parse_args()

    dec = Decoder(a.model_dir, a.threads, a.profile, share_kv=not a.no_share_kv, max_kv=a.max_kv)
    ids = dec.tok(a.prompt)["input_ids"]
    logger.info(f"{a.model_dir} | backend {dec.backend} (contract {dec.contract}) | "
                f"{dec.n_layers}L kv_heads={dec.kv_heads} head_size={dec.head_size} | "
                f"threads={a.threads} | temp={a.temperature} | share_kv={dec.share_kv}")

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
