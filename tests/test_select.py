"""Unit tests for token selection (the temperature / contract branching), no model needed.
_select takes the session output dict; contract routes A (full logits), H (shortlist), B (token)."""
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


# --- contract A: full (1,V) logits ---
def test_greedy_picks_argmax():
    assert _dec("A")._select(_logits(50, {7: 2.0, 13: 5.0}), 0.0, None) == 13


def test_sampling_stays_within_scored_candidates():
    """temp>0 must never sample a -1e9-filled (unscored) index — probed-softmax correctness."""
    cand = [3, 17, 42, 88]
    od = _logits(100, dict.fromkeys(cand, 1.0))
    rng = np.random.default_rng(0)
    picks = {_dec("A")._select(od, 1.0, rng) for _ in range(300)}
    assert picks <= set(cand)


def test_sampling_is_seed_reproducible():
    od = _logits(100, {3: 1.0, 17: 0.5, 42: 2.0, 88: 0.1})
    a = _dec("A")._select(od, 0.8, np.random.default_rng(123))
    b = _dec("A")._select(od, 0.8, np.random.default_rng(123))
    assert a == b


# --- contract H: fused-op candidate shortlist (cand_logits, cand_ids) ---
def _shortlist(pairs):  # {token_id: logit}
    ids = np.array([list(pairs)], np.int64)
    lg = np.array([list(pairs.values())], np.float32)
    return {"cand_logits": lg, "cand_ids": ids}


def test_contract_h_greedy_returns_best_token_id():
    assert _dec("H")._select(_shortlist({41: 0.1, 7: 9.0, 88: 2.0}), 0.0, None) == 7


def test_contract_h_sampling_within_shortlist():
    pairs = {3: 1.0, 17: 0.5, 42: 2.0, 88: 0.1}
    rng = np.random.default_rng(0)
    picks = {_dec("H")._select(_shortlist(pairs), 1.0, rng) for _ in range(300)}
    assert picks <= set(pairs)


# --- contract B: token-out, greedy only ---
def test_contract_b_greedy_returns_token_id():
    d = _dec("B", token_out="next_token")
    assert d._select({"next_token": np.array([[0, 0, 41]])}, 0.0, None) == 41


def test_contract_b_rejects_temperature():
    d = _dec("B", token_out="next_token")
    with pytest.raises(ValueError):
        d._select({"next_token": np.array([[5]])}, 0.8, np.random.default_rng(0))
