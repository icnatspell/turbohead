"""Minimal raw-ORT encoder-decoder decode loop + per-step bench for genai-exported Whisper.

This is the runtime harness `decode_loop.py` doesn't cover (it rejects Whisper: cross-attn KV has no
`present_*` output, input_ids is int32, there's an encoder pass and no attention_mask). Standalone so
core stays text-LM-only until this graduates.

Drives encoder.onnx once (audio -> hidden_states + constant cross-KV) then loops decoder.onnx (self-KV
grows past->present, cross-KV stays fixed). Handles all three head shapes like the core loop:
  - dense / onnx-flash: graph emits `logits` (1,1,V) -> argmax in numpy
  - fused-flash:        graph emits `cand_logits`/`cand_ids` shortlist -> argmax over candidates

    uv run python experimental/whisper_head/whisper_decode.py <export_dir> [--decoder decoder.onnx]
                       [--clip 0] [--max-new 64] [--bench] [--reps 5]

`<export_dir>` is a genai whisper export (encoder.onnx + decoder.onnx + configs). `--decoder` picks the
decoder file inside it (e.g. a spliced `model_fused.onnx`). `--bench` times steady-state decoder steps.
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from loguru import logger


def load_clip(idx):
    import soundfile as sf
    from datasets import Audio, load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation").cast_column("audio", Audio(decode=False))
    wav, sr = sf.read(io.BytesIO(ds[idx]["audio"]["bytes"]))
    return wav, sr


def session(path, lib=None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # match the core bench: head is memory-bound, best single-threaded
    if lib:
        so.register_custom_ops_library(str(lib))
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class Whisper:
    """encoder once, then a decoder step loop. `decoder_file` may be dense or a spliced variant."""

    def __init__(self, export_dir, decoder_file="decoder.onnx"):
        self.dir = Path(export_dir)
        lib = self.dir / "libturbohead.so"  # splice copies it next to the spliced model
        dpath = self.dir / decoder_file
        self.enc = session(self.dir / "encoder.onnx")
        self.dec = session(dpath, lib if lib.exists() and decoder_file != "decoder.onnx" else None)
        self.nlayers = sum(o.name.startswith("present_key_self_") for o in self.dec.get_outputs())
        self.heads, self.hsize = self._self_kv_shape()
        outs = {o.name for o in self.dec.get_outputs()}
        self.shortlist = "cand_ids" in outs  # fused backend
        # int4 CPU build is fp32 io; read the actual dtype off the encoder output to be safe.
        self.kv_dt = np.float16 if "float16" in self.enc.get_outputs()[1].type else np.float32

    def _self_kv_shape(self):
        for i in self.dec.get_inputs():
            if i.name == "past_key_self_0":
                return int(i.shape[1]), int(i.shape[3])  # num_heads, head_size
        raise RuntimeError("no past_key_self_0 input")

    def encode(self, feats):
        """audio_features (1,mel,3000) -> {past_key_cross_i, past_value_cross_i} (constant for the clip)."""
        out = self.enc.run(None, {"audio_features": feats.astype(self.kv_dt)})
        names = [o.name for o in self.enc.get_outputs()]
        d = dict(zip(names, out))
        cross = {}
        for i in range(self.nlayers):
            cross[f"past_key_cross_{i}"] = d[f"present_key_cross_{i}"]
            cross[f"past_value_cross_{i}"] = d[f"present_value_cross_{i}"]
        return cross

    def _empty_self(self):
        z = np.zeros((1, self.heads, 0, self.hsize), self.kv_dt)
        return {f"past_{k}_self_{i}": z for i in range(self.nlayers) for k in ("key", "value")}

    def step(self, ids, self_kv, cross):
        feeds = {"input_ids": np.asarray(ids, np.int32).reshape(1, -1), **self_kv, **cross}
        out = self.dec.run(None, feeds)
        od = dict(zip([o.name for o in self.dec.get_outputs()], out))
        nxt = {f"past_{k}_self_{i}": od[f"present_{k}_self_{i}"]
               for i in range(self.nlayers) for k in ("key", "value")}
        if self.shortlist:
            tok = int(od["cand_ids"].ravel()[od["cand_logits"].ravel().argmax()])
        else:
            tok = int(od["logits"][0, -1].argmax())
        return tok, nxt, od

    def generate(self, feats, prompt_ids, eos, max_new=64):
        cross = self.encode(feats)
        self_kv = self._empty_self()
        out, step_ids, times = [], list(prompt_ids), []  # prefill prefix, then one token/step
        for _ in range(max_new):
            t = time.perf_counter()
            tok, self_kv, _ = self.step(step_ids, self_kv, cross)
            times.append(time.perf_counter() - t)
            out.append(tok)
            if tok == eos:
                break
            step_ids = [tok]
        return out, times

    def bench(self, feats, prompt_ids, steps=48):
        """Steady-state per-step time, teacher-forced over a FIXED token sequence so dense and flash run
        byte-identical compute except the head — isolating the head cost (the thesis). Token *value*
        doesn't affect step cost; we force the last prompt id so KV grows the same for every variant."""
        cross = self.encode(feats)
        self_kv = self._empty_self()
        _, self_kv, _ = self.step(prompt_ids, self_kv, cross)  # prefill (not timed)
        forced = prompt_ids[-1]
        times = []
        for _ in range(steps):
            t = time.perf_counter()
            _, self_kv, _ = self.step([forced], self_kv, cross)
            times.append(time.perf_counter() - t)
        return times


def whisper_prompt(model_id):
    """forced decoder prefix + eos via the HF processor (transcribe, no timestamps). offline-only dep.
    `model_id` is the HF hub id (genai exports drop `audio_processor_config.json`, not the transformers
    `preprocessor_config.json` the processor wants — the mel/tokenizer are identical to the hub model)."""
    from transformers import WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_id)
    forced = proc.get_decoder_prompt_ids(language="en", task="transcribe")  # [(pos, id), ...]
    start = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    prompt = [start] + [i for _, i in forced]
    return proc, prompt, proc.tokenizer.eos_token_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export_dir")
    ap.add_argument("--model", default="", help="HF id for the processor; default reads "
                    "export_dir/source_model.txt (written by the build driver)")
    ap.add_argument("--decoder", default="decoder.onnx", help="decoder file inside export_dir")
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    model_id = a.model or (Path(a.export_dir) / "source_model.txt").read_text().strip()
    proc, prompt, eos = whisper_prompt(model_id)
    wav, sr = load_clip(a.clip)
    feats = proc(wav, sampling_rate=sr, return_tensors="np").input_features
    w = Whisper(a.export_dir, a.decoder)
    logger.info(f"{a.decoder}: {w.nlayers} layers, {w.heads}h x {w.hsize}, "
                f"{'shortlist' if w.shortlist else 'logits'}-out, prefix {prompt}")

    out, times = w.generate(feats, prompt, eos, a.max_new)
    text = proc.tokenizer.decode(out, skip_special_tokens=True)
    logger.info(f"transcript: {text!r}")
    logger.info(f"{len(out)} tokens, median step {np.median(times)*1e3:.2f} ms")

    if a.bench:
        per = [np.median(w.bench(feats, prompt, a.max_new)) for _ in range(a.reps)]
        logger.info(f"BENCH {a.decoder}: median decoder step {np.median(per)*1e3:.3f} ms "
                    f"(teacher-forced steady-state, {a.reps} reps, 1 thread)")


if __name__ == "__main__":
    main()
