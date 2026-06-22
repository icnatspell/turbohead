"""Bench every head variant for each built Whisper model -> one speed table (the headline result).

Runs each (model, variant) in its OWN subprocess: two custom-op `.so` in one process segfault (same
reason `turbohead-bench` forks per model). Teacher-forced steady-state decoder step, 1 thread.

    uv run python experimental/whisper_head/sweep.py [whisper_tiny whisper_base whisper_small]

Variants per model: dense-int4 (genai default head), dense-fp32 (the RESULTS.md headline baseline),
onnx-flash, fused-flash. Speedups are reported vs BOTH dense baselines, mirroring RESULTS.md.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("artifacts")
DECODE = ["uv", "run", "python", "experimental/whisper_head/whisper_decode.py"]
# (label, export_subdir, decoder_file). Four dense-head precisions (int4 body throughout) + two flash.
VARIANTS = [
    ("fp32", "export", "decoder_f32.onnx"),
    ("fp16", "head16", "model.onnx"),
    ("int8", "head8g128", "model.onnx"),
    ("int4", "head4g128", "model.onnx"),
    ("flash", "export_onnx", "model.onnx"),
    ("flash-fused", "export_fused", "model.onnx"),
]


def bench_one(slug, sub, dec, reps, steps):
    d = ROOT / slug / sub
    if not (d / dec).exists():
        return None
    out = subprocess.run(DECODE + [str(d), "--decoder", dec, "--bench",
                                   "--reps", str(reps), "--max-new", str(steps)],
                         capture_output=True, text=True).stderr
    m = re.search(r"median decoder step ([\d.]+) ms", out)
    return float(m.group(1)) if m else None


def main():
    slugs = sys.argv[1:] or ["whisper_tiny", "whisper_base", "whisper_small"]
    labels = [lbl for lbl, _, _ in VARIANTS]

    def ms(x):
        return f"{x:.2f}" if x else "  -"

    print("\n== median decoder step (ms), int4 body, teacher-forced steady-state, 1 thread ==")
    print(f"{'model':<14}" + "".join(f"{lab:>13}" for lab in labels))
    print("-" * (14 + 13 * len(labels)))
    rows = {}
    for slug in slugs:
        t = {lbl: bench_one(slug, sub, dec, 5, 48) for lbl, sub, dec in VARIANTS}
        rows[slug] = t
        if not any(t.values()):
            print(f"{slug:<14}  (not built — run build_whisper.py)")
            continue
        print(f"{slug:<14}" + "".join(f"{ms(t[lab]):>13}" for lab in labels))

    print("\n== flash-fused speedup vs each dense-head precision (>1x = flash faster) ==")
    print(f"{'model':<14}{'vs fp32':>12}{'vs fp16':>12}{'vs int8':>12}{'vs int4':>12}")
    print("-" * 62)
    for slug in slugs:
        t = rows.get(slug, {})
        f = t.get("flash-fused")
        if not f:
            continue

        def sp(base, t=t, f=f):
            return f"{t[base]/f:.2f}x" if t.get(base) else "  -"
        print(f"{slug:<14}{sp('fp32'):>12}{sp('fp16'):>12}{sp('int8'):>12}{sp('int4'):>12}")
    print()


if __name__ == "__main__":
    main()
