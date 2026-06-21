"""Integration: run the real spliced model end-to-end. Skipped if the artifact isn't built
(e.g. in CI) — build it via the README surgery steps to exercise this."""
import os
import numpy as np
import pytest
from turbohead.inference.decode_loop import Decoder

MODEL = "artifacts/qwen3_0_6b/onnx"    # logits-out (onnx backend)
FUSED = "artifacts/qwen3_0_6b/fused"   # shortlist-out (fused custom op)
HYBRID = "artifacts/lfm2_5_350m/fused"  # hybrid (conv + sparse-index attention) — generic state path
EMBEDS = "artifacts/qwen3_5_0_8b/fused"  # embeds-in (split embedding) + 3-D M-RoPE position_ids


def test_zero_state_seeds_kv_and_recurrent_shapes():
    """The generic state seed: batch dim -> 1, symbolic seq dim -> 0 (KV grows), concrete dims
    kept (recurrent/conv state seeds full-size). Dtype follows the input. No model needed."""
    class Inp:  # minimal stand-in for an ORT input descriptor
        def __init__(self, type, shape):
            self.type, self.shape = type, shape

    kv = Decoder._zero_state(Inp("tensor(float)", ["batch", 8, "past_seq", 96]))
    assert kv.shape == (1, 8, 0, 96) and kv.dtype == np.float32
    rec = Decoder._zero_state(Inp("tensor(float)", ["batch", 16, 128, 128]))
    assert rec.shape == (1, 16, 128, 128)
    conv = Decoder._zero_state(Inp("tensor(float16)", ["batch", 6144, 3]))
    assert conv.shape == (1, 6144, 3) and conv.dtype == np.float16


@pytest.mark.skipif(not os.path.isdir(HYBRID), reason=f"{HYBRID} not built")
def test_hybrid_model_decodes():
    """Hybrid model (interleaved conv + sparse-index attention layers) decodes via the generic
    state path — regression guard for the past_conv.* / past_key_values.N.* seeding + remap."""
    dec = Decoder(HYBRID, threads=1)
    out, tps = dec.generate(dec.tok("Once upon a time,")["input_ids"], max_new=8)
    assert out and tps > 0


@pytest.mark.skipif(not os.path.isdir(EMBEDS), reason=f"{EMBEDS} not built")
def test_embeds_in_model_decodes():
    """inputs_embeds graph (tied-embedding lookup done in numpy from head_W) + 3-D M-RoPE
    position_ids decodes — regression guard for the embeds-in feed path."""
    dec = Decoder(EMBEDS, threads=1)
    assert dec.embeds_in and dec.pos_rank == 3
    out, tps = dec.generate(dec.tok("Once upon a time,")["input_ids"], max_new=8)
    assert out and tps > 0


@pytest.fixture(scope="module")
def dec():
    if not os.path.isdir(MODEL):
        pytest.skip(f"{MODEL} not built (run the surgery pipeline)")
    return Decoder(MODEL, threads=1)


@pytest.mark.skipif(not os.path.isdir(FUSED), reason=f"{FUSED} not built (turbohead-splice --backend fused)")
def test_fused_greedy_matches_onnx(dec):
    """The fused custom-op kernel (shortlist-out) must reproduce the onnx logits-out graph exactly.
    Gate measured 100% over 12 prompts x 128 tokens incl. -ffast-math; a couple here guard it."""
    fused = Decoder(FUSED, threads=1)
    for p in ("Once upon a time, in a small village,", "def fibonacci(n):"):
        ids = dec.tok(p)["input_ids"]
        assert dec.generate(ids, max_new=32)[0] == fused.generate(ids, max_new=32)[0]


def test_greedy_is_deterministic_and_nonempty(dec):
    ids = dec.tok("Once upon a time,")["input_ids"]
    out1, tps = dec.generate(ids, max_new=8)
    out2, _ = dec.generate(ids, max_new=8)
    assert out1 and out1 == out2   # greedy: same prompt -> same tokens
    assert tps > 0


def test_sampling_seed_reproducible(dec):
    ids = dec.tok("Once upon a time,")["input_ids"]
    a, _ = dec.generate(ids, max_new=8, temperature=0.8, seed=1)
    b, _ = dec.generate(ids, max_new=8, temperature=0.8, seed=1)
    assert a == b


# --- shared-KV sliding window (fixed max_kv): generate past the cap by dropping old context ---
def test_reprefill_resets_shared_cache(dec):
    """The eviction mechanism: re-prefilling a short window B into a buffer that held longer text A
    must give byte-identical head output to a FRESH prefill of B — proving the slide overwrites
    from slot 0 and the stale A slots beyond B are masked out (no leakage, no per-model RoPE)."""
    a = dec.tok("The quick brown fox jumps over the lazy dog near the wide river bank.")["input_ids"]
    b = dec.tok("A short sentence.")["input_ids"]
    cap = len(a) + 4
    d1 = Decoder(MODEL, threads=1, max_kv=cap)
    d1._setup_shared(cap)
    d1._step_shared(np.array([a], np.int64), len(a), 0)         # fill buffer with A
    re = d1._step_shared(np.array([b], np.int64), len(b), 0)    # slide: re-prefill B over A
    d2 = Decoder(MODEL, threads=1, max_kv=cap)
    d2._setup_shared(cap)
    fresh = d2._step_shared(np.array([b], np.int64), len(b), 0)
    for k in re:
        assert np.array_equal(re[k], fresh[k]), k


def test_sliding_window_generates_past_cap(dec):
    """A fixed max_kv smaller than prompt+max_new must keep generating (slide), not stop early."""
    ids = dec.tok("Once upon a time, in a small village, there lived")["input_ids"]
    cap = len(ids) + 8                                          # only 8 spare decode slots
    out, tps = Decoder(MODEL, threads=1, max_kv=cap).generate(ids, max_new=40)
    assert len(out) > cap - len(ids) and tps > 0               # went past the buffer's spare room


def test_sliding_prefix_matches_full_context(dec):
    """Until the buffer first fills, the sliding decoder still holds the whole context, so its
    greedy output is identical to a full-context run; they diverge only once eviction starts."""
    ids = dec.tok("Once upon a time, in a small village,")["input_ids"]
    cap = len(ids) + 8
    full = dec.generate(ids, max_new=40)[0]                     # dec: no max_kv -> never slides
    slid = Decoder(MODEL, threads=1, max_kv=cap).generate(ids, max_new=40)[0]
    pre = cap - len(ids)                                        # tokens emitted before the first slide
    assert slid[:pre] == full[:pre]
