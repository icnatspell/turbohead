"""Integration: run the real spliced model end-to-end. Skipped if the artifact isn't built
(e.g. in CI) — build it via the README surgery steps to exercise this."""
import os
import pytest
from turbohead.inference.decode_loop import Decoder

MODEL = "artifacts/qwen3_0_6b_flash"
FUSED = "artifacts/qwen3_0_6b_fused"
pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL),
                                reason=f"{MODEL} not built (run the surgery pipeline)")


@pytest.fixture(scope="module")
def dec():
    return Decoder(MODEL, threads=1)


@pytest.mark.skipif(not os.path.isdir(FUSED), reason=f"{FUSED} not built (run splice_fused)")
def test_fused_greedy_matches_contract_a(dec):
    """The fused custom-op kernel (contract H) must reproduce the contract-A graph exactly.
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
