"""Integration: run the real spliced model end-to-end. Skipped if the artifact isn't built
(e.g. in CI) — build it via the README surgery steps to exercise this."""
import os
import pytest
from turbohead.inference.decode_loop import Decoder

MODEL = "artifacts/qwen3_0_6b_flash"
pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL),
                                reason=f"{MODEL} not built (run the surgery pipeline)")


@pytest.fixture(scope="module")
def dec():
    return Decoder(MODEL, threads=1)


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
