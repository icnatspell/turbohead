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


def test_balanced_assign_forbid_excludes_primary():
    """The 2nd-home layer (--r 2) keeps exact capacity AND never re-homes a token in its primary."""
    W, C, cap, K = _toy()
    primary = balanced_assign(W, C, cap)
    second = balanced_assign(W, C, cap, forbid=primary)
    assert (np.bincount(second, minlength=K) == cap).all()  # still balanced
    assert (second != primary).all()                        # 2nd home is a different cluster


def test_build_r2_table_shape_and_primary_cnorm():
    """r=2 build: table is (K, 2cap), each layer a full permutation, Cnorm from PRIMARY members only."""
    W, C, cap, K = _toy()
    D = W.shape[1]
    primary = balanced_assign(W, C, cap)
    second = balanced_assign(W, C, cap, forbid=primary)
    homes = np.stack([primary, second], axis=1)
    Cnorm, Wperm, Vmap = build(W, homes, cap)
    assert Wperm.shape == (K, 2 * cap, D) and Vmap.shape == (K, 2 * cap)
    # each layer (cap-wide block) is its own full permutation of the vocab
    for j in range(2):
        block = Vmap[:, j * cap : (j + 1) * cap]
        assert sorted(block.reshape(-1).tolist()) == list(range(W.shape[0]))
    # Cnorm must equal the r=1 (primary-only) Cnorm — the 2nd layer must NOT blur it
    Cnorm1, _, _ = build(W, primary, cap)
    np.testing.assert_array_equal(Cnorm, Cnorm1)
