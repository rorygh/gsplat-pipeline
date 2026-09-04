"""CPU-only tests for the SuGaR-lite geometry helpers (no checkpoint/GPU)."""

from __future__ import annotations

import torch

from gsplat_pipeline.sugar import _depth_to_normal, _quat_to_rotmat, _thin_axis


def test_quat_to_rotmat_identity():
    q = torch.tensor([[1.0, 0, 0, 0]])
    torch.testing.assert_close(_quat_to_rotmat(q)[0], torch.eye(3))


def test_quat_to_rotmat_is_rotation():
    q = torch.randn(8, 4)
    R = _quat_to_rotmat(q)
    torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(3).expand(8, 3, 3), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(torch.det(R), torch.ones(8), atol=1e-5, rtol=1e-4)


def test_thin_axis_picks_smallest_scale_direction():
    # identity rotation -> axes are world x/y/z; smallest scale on axis 1 (y)
    q = torch.tensor([[1.0, 0, 0, 0]])
    scales = torch.tensor([[1.0, 0.01, 1.0]])
    n = _thin_axis(q, scales)[0]
    assert n.abs().argmax().item() == 1
    torch.testing.assert_close(n.abs(), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0)


def test_depth_to_normal_of_a_frontal_plane():
    # a plane at constant depth faces the camera: normal ~ (0, 0, +/-1)
    h = w = 32
    depth = torch.full((h, w), 3.0)
    K = torch.tensor([[40.0, 0, w / 2], [0, 40.0, h / 2], [0, 0, 1.0]])
    n = _depth_to_normal(depth, K)
    interior = n[2:-2, 2:-2]
    assert interior[..., 2].abs().mean() > 0.99
    assert interior[..., :2].abs().mean() < 0.02
