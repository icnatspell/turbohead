"""Speed matrix — decode tok/s for a set of models across thread counts, median ± std over reps.

Single runs are noise (±3-5 tok/s); this repeats each config and reports median±std. Each model
runs in its **own subprocess**: a process may load at most one turbohead custom-op .so (two in one
process segfaults — see docs/ORT_QUIRKS.md), and per-process isolation also keeps ORT thread pools
and arenas from leaking between configs.

  driver  — loops (model x threads), spawns a worker each, collects JSON, prints the matrix.
  worker  — `--worker DIR`: load one model, 1 warmup + R timed reps, print one JSON line of tok/s.

Usage:
  uv run turbohead-bench artifacts/qwen3_0_6b/onnx artifacts/qwen3_0_6b/fused \\
      --threads 1,2,4,8 --reps 7 [--temperature 0.8] [--max-new 64]
"""
import sys
import json
import argparse
import subprocess
import statistics
from loguru import logger

DEFAULT_PROMPT = "Once upon a time, in a small village,"


def _worker(args):
    """Run one model and emit {"tps": [...]} on stdout. Imports kept local to the worker path."""
    from turbohead.inference.decode_loop import Decoder
    dec = Decoder(args.worker, threads=args.threads)
    ids = dec.tok(args.prompt)["input_ids"]
    rates = []
    for r in range(args.reps + 1):  # 1 warmup
        _, tps = dec.generate(ids, args.max_new, temperature=args.temperature, seed=args.seed)
        if r > 0:
            rates.append(tps)
    print(json.dumps({"tps": rates, "backend": dec.backend, "contract": dec.contract}))


def bench_one(model, threads, reps, max_new, prompt, temperature, seed):
    """Spawn a worker subprocess for (model, threads); return its rates list (or [] on failure)."""
    cmd = [sys.executable, "-m", "turbohead.eval.benchmark", "--worker", model,
           "--threads", str(threads), "--reps", str(reps), "--max-new", str(max_new),
           "--prompt", prompt, "--temperature", str(temperature)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        logger.error(f"{model} @ {threads}t failed (rc={p.returncode}):\n{p.stderr[-500:]}")
        return None
    return json.loads(p.stdout.strip().splitlines()[-1])  # last line = our JSON


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", help="model dirs (driver mode)")
    ap.add_argument("--worker", help="internal: run this single model and emit JSON")
    ap.add_argument("--threads", default="1,2,4,8", help="comma-separated thread counts")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    if a.worker:
        a.threads = int(a.threads.split(",")[0])
        return _worker(a)

    threads = [int(t) for t in a.threads.split(",")]
    logger.info(f"{len(a.models)} models x {threads} threads | reps={a.reps} "
                f"max_new={a.max_new} temp={a.temperature}")
    rows = {}  # model -> {threads: (median, std, backend)}
    for m in a.models:
        rows[m] = {}
        for t in threads:
            res = bench_one(m, t, a.reps, a.max_new, a.prompt, a.temperature, a.seed)
            if res:
                r = res["tps"]
                med, std = statistics.median(r), (statistics.stdev(r) if len(r) > 1 else 0.0)
                rows[m][t] = (med, std, res["backend"])
                logger.info(f"  {m.split('/')[-1]:32s} {t}t  {med:6.1f} ± {std:4.1f} tok/s "
                            f"({res['backend']})")

    # Matrix: tok/s median±std, normalized to the first model (baseline) per thread count.
    hdr = "model".ljust(34) + "".join(f"{t}t".rjust(16) for t in threads)
    logger.info("\n" + hdr)
    logger.info("-" * len(hdr))
    base = a.models[0]
    for m in a.models:
        cells = []
        for t in threads:
            if t not in rows[m]:
                cells.append("—".rjust(16))
                continue
            med, std, _ = rows[m][t]
            spd = med / rows[base][t][0] if t in rows[base] else float("nan")
            cells.append(f"{med:5.1f}±{std:3.1f}({spd:.2f}x)".rjust(16))
        logger.info(m.split("/")[-1].ljust(34) + "".join(cells))
    logger.info(f"\n(speedup x vs {base.split('/')[-1]}, per thread count)")


if __name__ == "__main__":
    main()
