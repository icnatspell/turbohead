"""Unit tests for token selection (the temperature / output-shape branching), no model needed.
_select takes the session output dict; routes on the graph output: logits-out (full logits),
shortlist-out (fused candidate shortlist), token-out (the id directly)."""
import numpy as np
import pytest
from turbohead.inference.decode_loop import Decoder


def _dec(contract, token_out="logits"):
    d = Decoder.__new__(Decoder)   # bypass __init__: _select only touches self.contract/token_out
    d.contract = contract
    d.token_out = token_out
    return d


def _logits(V, candidates, fill=-1e9):
    row = np.full((1, 1, V), fill, np.float32)
    for idx, val in candidates.items():
        row[0, 0, idx] = val
    return {"logits": row}


# --- logits-out: full (1,V) logits ---
def test_greedy_picks_argmax():
    assert _dec("logits-out")._select(_logits(50, {7: 2.0, 13: 5.0}), 0.0, None) == 13


def test_sampling_stays_within_scored_candidates():
    """temp>0 must never sample a -1e9-filled (unscored) index — probed-softmax correctness."""
    cand = [3, 17, 42, 88]
    od = _logits(100, dict.fromkeys(cand, 1.0))
    rng = np.random.default_rng(0)
    picks = {_dec("logits-out")._select(od, 1.0, rng) for _ in range(300)}
    assert picks <= set(cand)


def test_sampling_is_seed_reproducible():
    od = _logits(100, {3: 1.0, 17: 0.5, 42: 2.0, 88: 0.1})
    a = _dec("logits-out")._select(od, 0.8, np.random.default_rng(123))
    b = _dec("logits-out")._select(od, 0.8, np.random.default_rng(123))
    assert a == b


# --- shortlist-out: fused-op candidate shortlist (cand_logits, cand_ids) ---
def _shortlist(pairs):  # {token_id: logit}
    ids = np.array([list(pairs)], np.int64)
    lg = np.array([list(pairs.values())], np.float32)
    return {"cand_logits": lg, "cand_ids": ids}


def test_shortlist_greedy_returns_best_token_id():
    assert _dec("shortlist-out")._select(_shortlist({41: 0.1, 7: 9.0, 88: 2.0}), 0.0, None) == 7


def test_shortlist_sampling_within_shortlist():
    pairs = {3: 1.0, 17: 0.5, 42: 2.0, 88: 0.1}
    rng = np.random.default_rng(0)
    picks = {_dec("shortlist-out")._select(_shortlist(pairs), 1.0, rng) for _ in range(300)}
    assert picks <= set(pairs)


def test_shortlist_sampling_dedups_duplicate_ids():
    """multiple-assignment (--r 2) can list a token twice (it sat in two probed clusters) with equal
    logits. Sampling must weight it ONCE, not by its multiplicity. Here id 3 appears twice and id 7
    once, all equal logit -> each should win ~1/2; without dedup id 3 would win ~2/3."""
    od = {"cand_logits": np.zeros((1, 3), np.float32),
          "cand_ids": np.array([[3, 3, 7]], np.int64)}
    rng = np.random.default_rng(0)
    picks = np.array([_dec("shortlist-out")._select(od, 1.0, rng) for _ in range(4000)])
    frac3 = (picks == 3).mean()
    assert 0.45 < frac3 < 0.55, frac3   # ~0.5 deduped; ~0.667 if the duplicate is double-counted


# --- token-out: the id directly, greedy only ---
def test_tokenout_greedy_returns_token_id():
    d = _dec("token-out", token_out="next_token")
    assert d._select({"next_token": np.array([[0, 0, 41]])}, 0.0, None) == 41


def test_tokenout_rejects_temperature():
    d = _dec("token-out", token_out="next_token")
    with pytest.raises(ValueError):
        d._select({"next_token": np.array([[5]])}, 0.8, np.random.default_rng(0))


# --- shared-KV sliding window: bookkeeping only, no model ---
@pytest.mark.parametrize("cap", [2, 3, 4, 8, 16, 128, 2048])
def test_slide_keep_leaves_headroom(cap):
    """The retained-window size must leave >=1 free slot after a slide (else infinite slide / OOB)
    and keep a useful chunk of recent context."""
    keep = Decoder._slide_keep(cap)
    assert 1 <= keep <= cap - 1            # in-bounds write + room for the next token
    if cap >= 8:
        assert keep >= cap // 2            # retains a useful recent window


def test_sliding_schedule_never_writes_past_buffer():
    """Simulate generate's shared-KV write schedule (the OOB-critical part): prefill n0, then
    single-token steps into a cap buffer that slides at overflow. The in-buffer length must never
    exceed cap (an over-cap write is an out-of-bounds segfault) and must slide at least once."""
    cap, n0 = 16, 5
    clen, lens = n0, [n0]
    for _ in range(100):
        clen = clen + 1 if clen + 1 <= cap else Decoder._slide_keep(cap)
        lens.append(clen)
    assert max(lens) <= cap                                  # no write past the buffer
    assert min(lens) >= 1
    assert any(b < a for a, b in zip(lens, lens[1:]))        # slid at least once
