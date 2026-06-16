"""Unit tests for token selection (the temperature / contract branching), no model needed."""
import numpy as np
import pytest
from turbohead.inference.decode_loop import Decoder


def _dec(contract):
    d = Decoder.__new__(Decoder)   # bypass __init__: _select only touches self.contract
    d.contract = contract
    return d


def _logits(V, candidates, fill=-1e9):
    row = np.full((1, 1, V), fill, np.float32)
    for idx, val in candidates.items():
        row[0, 0, idx] = val
    return row


def test_greedy_picks_argmax():
    d = _dec("A")
    assert d._select(_logits(50, {7: 2.0, 13: 5.0}), 0.0, None) == 13


def test_sampling_stays_within_scored_candidates():
    """temp>0 must never sample a -1e9-filled (unscored) index — probed-softmax correctness."""
    d = _dec("A")
    cand = [3, 17, 42, 88]
    row = _logits(100, dict.fromkeys(cand, 1.0))
    rng = np.random.default_rng(0)
    picks = {d._select(row, 1.0, rng) for _ in range(300)}
    assert picks <= set(cand)


def test_sampling_is_seed_reproducible():
    d = _dec("A")
    row = _logits(100, {3: 1.0, 17: 0.5, 42: 2.0, 88: 0.1})
    a = d._select(row, 0.8, np.random.default_rng(123))
    b = d._select(row, 0.8, np.random.default_rng(123))
    assert a == b


def test_contract_b_greedy_returns_token_id():
    d = _dec("B")
    assert d._select(np.array([[0, 0, 41]]), 0.0, None) == 41


def test_contract_b_rejects_temperature():
    d = _dec("B")
    with pytest.raises(ValueError):
        d._select(np.array([[5]]), 0.8, np.random.default_rng(0))
