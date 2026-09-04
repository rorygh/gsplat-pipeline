"""Fast, CPU-only tests for the mask cleanup + temporal-stabilization helpers
(no rembg / transformers / model downloads needed)."""

from __future__ import annotations

import numpy as np

from gsplat_pipeline.masking import (
    _clean_binary,
    _compose,
    _fill_holes,
    _largest_component,
    _stabilize,
    _warp,
)


def test_largest_component_drops_speckle():
    m = np.zeros((40, 40), bool)
    m[5:25, 5:25] = True  # the object
    m[0, 0] = True  # a speckle
    m[38:40, 38:40] = True  # another speckle
    out = _largest_component(m)
    assert out[10, 10] and not out[0, 0] and not out[39, 39]


def test_fill_holes():
    m = np.zeros((30, 30), bool)
    m[5:25, 5:25] = True
    m[12:18, 12:18] = False  # a hole punched in the middle
    out = _fill_holes(m)
    assert out[15, 15]
    assert not out[2, 2]  # outside stays background


def test_fill_holes_object_touching_corner():
    m = np.zeros((30, 30), bool)
    m[0:20, 0:20] = True  # object runs into the top-left corner
    m[5:12, 5:12] = False  # enclosed hole
    out = _fill_holes(m)
    assert out[8, 8]  # hole filled
    assert not out[25, 25]  # outside background NOT filled


def test_clean_binary_empty_input():
    prob = np.zeros((10, 10), np.float32)
    assert not _clean_binary(prob, 0.5).any()


def test_warp_identity():
    img = np.random.default_rng(0).random((16, 16), dtype=np.float32)
    flow = np.zeros((16, 16, 2), np.float32)
    np.testing.assert_allclose(_warp(img, flow), img, atol=1e-5)


def test_warp_translation():
    img = np.zeros((20, 20), np.float32)
    img[10, 10] = 1.0
    flow = np.zeros((20, 20, 2), np.float32)
    flow[..., 0] = 2.0  # sample 2px to the right -> feature appears 2px left
    out = _warp(img, flow)
    assert out[10, 8] > 0.5


def test_compose_adds_translations():
    flow_ab = np.full((8, 8, 2), 0.0, np.float32)
    flow_ab[..., 0] = 1.0
    flow_bc = np.full((8, 8, 2), 0.0, np.float32)
    flow_bc[..., 0] = 3.0
    np.testing.assert_allclose(_compose(flow_ab, flow_bc)[..., 0], 4.0, atol=1e-5)


def _translating_sequence(n=5, shift=2, size=64):
    """A textured frame translating right by `shift` px/frame, with a square
    mask riding along."""
    rng = np.random.default_rng(42)
    base = rng.random((size, size * 2), dtype=np.float32)
    grays, probs = [], []
    for i in range(n):
        x0 = 10 + i * shift
        g = (base[:, x0:x0 + size] * 255).astype(np.uint8)
        p = np.zeros((size, size), np.float32)
        p[20:44, 15:39] = 1.0
        grays.append(g)
        probs.append(p)
    return grays, probs


def test_stabilize_noop_when_short():
    grays, probs = _translating_sequence(n=2)
    assert _stabilize(probs, grays, window=2, area_gate=0.5) is probs


def test_stabilize_repairs_dropout():
    grays, probs = _translating_sequence(n=5)
    good_area = float((probs[2] >= 0.5).mean())
    probs[2] = np.zeros_like(probs[2])  # model dropped the object on frame 2

    out = _stabilize(probs, grays, window=2, area_gate=0.5)
    repaired_area = float((out[2] >= 0.5).mean())

    assert repaired_area > 0.5 * good_area  # neighbours filled it back in
    # and it lands roughly where the object actually is on frame 2
    ys, xs = np.where(out[2] >= 0.5)
    assert 18 <= ys.mean() <= 46 and 13 <= xs.mean() <= 41


def test_stabilize_preserves_good_sequence():
    grays, probs = _translating_sequence(n=5)
    before = [float((p >= 0.5).mean()) for p in probs]
    out = _stabilize(probs, grays, window=2, area_gate=0.5)
    after = [float((p >= 0.5).mean()) for p in out]
    for b, a in zip(before, after):
        assert abs(a - b) < 0.15  # stable frames stay put
