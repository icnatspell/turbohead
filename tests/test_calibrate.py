"""Unit test for lever-4 calibration ranking (numpy-only, no model needed)."""
import numpy as np
from turbohead.surgery.calibrate_misses import most_missed


def test_most_missed_ranks_chronic_misses_first():
    """A position is a miss when its true cluster ranks below P. most_missed returns the missed
    tokens by frequency. Here token 10 (cluster 2) is routed low 3x, token 30 (cluster 0) 1x,
    token 20 (cluster 1) always routed top -> never missed."""
    dense    = np.array([10, 10, 10, 20, 30])
    true_clu = np.array([2,  2,  2,  1,  0])      # cluster holding each dense token
    sims = np.array([[5, 5, 0],                   # cluster 2 lowest -> token 10 missed
                     [5, 5, 0],
                     [5, 5, 0],
                     [0, 9, 0],                   # cluster 1 top -> token 20 hit
                     [0, 9, 0]], np.float32)       # cluster 0 below cluster 1 -> token 30 missed
    out = most_missed(dense, true_clu, sims, P=1, top=10)
    assert out.tolist() == [10, 30]               # 10 missed 3x ranks before 30 (1x); 20 absent


def test_top_caps_the_list_and_dtype_is_int64():
    dense = np.array([1, 1, 2, 3])
    true_clu = np.array([0, 0, 0, 0])             # all true cluster 0
    sims = np.array([[0, 9], [0, 9], [0, 9], [0, 9]], np.float32)  # cluster 1 top -> cluster 0 all miss
    out = most_missed(dense, true_clu, sims, P=1, top=2)
    assert len(out) == 2 and out[0] == 1 and out.dtype == np.int64
