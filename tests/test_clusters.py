"""Unit tests for the balanced k-means surgery (numpy-only, no model/torch needed)."""
import numpy as np
from turbohead.surgery.build_clusters import balanced_assign, build


def _toy(V=40, D=8, cap=4, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((V, D)).astype(np.float32)
    K = V // cap
    C = W[rng.choice(V, K, replace=False)].copy()
    return W, C, cap, K


def test_balanced_assign_exact_capacity():
    """Every cluster ends with exactly `cap` tokens — the invariant splice/decode rely on."""
    W, C, cap, K = _toy()
    a = balanced_assign(W, C, cap)
    assert a.min() >= 0 and a.max() < K
    assert (np.bincount(a, minlength=K) == cap).all()


def test_build_shapes_and_vmap_is_a_permutation():
    W, C, cap, K = _toy()
    D = W.shape[1]
    a = balanced_assign(W, C, cap)
    Cnorm, Wperm, Vmap = build(W, a, cap)
    assert Cnorm.shape == (D, K)
    assert Wperm.shape == (K, cap, D)
    assert Vmap.shape == (K, cap)
    # each vocab id appears exactly once across all clusters (no drops, no dupes)
    assert sorted(Vmap.reshape(-1).tolist()) == list(range(W.shape[0]))
    # centroid columns are L2-normalized (fp16 stored -> loose tol)
    norms = np.linalg.norm(Cnorm.astype(np.float32), axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-2)
