"""CPU-only tests for wiring object masks into the data layer:
point-cloud filtering, per-view mask loading, and final-model pruning."""

from __future__ import annotations

import cv2
import numpy as np

from gsplat_pipeline.colmap.binary import Point3D
from gsplat_pipeline.colmap.dataset import (
    SceneData,
    _load_working_mask,
    _points_in_masks,
    _resolve_mask_paths,
    gaussians_in_masks,
)

W = H = 100
K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])


def _cam_at(z_back=5.0):
    """One camera on -Z looking toward +Z, so world +Z points into the image."""
    c2w = np.eye(4)
    c2w[2, 3] = -z_back
    return c2w[None]  # (1,4,4)


def _write_center_mask(path, box=30):
    m = np.zeros((H, W), np.uint8)
    c = W // 2
    m[c - box:c + box, c - box:c + box] = 255
    cv2.imwrite(str(path), m)


def test_resolve_mask_paths(tmp_path):
    (tmp_path / "frame_0001.png").write_bytes(b"x")
    paths = _resolve_mask_paths(tmp_path, ["frame_0001.jpg", "frame_0002.jpg"])
    assert paths[0] is not None and paths[1] is None
    assert _resolve_mask_paths(None, ["a"]) is None


def test_load_working_mask_resizes(tmp_path):
    p = tmp_path / "m.png"
    _write_center_mask(p)
    m = _load_working_mask(p, 50, 50, None)
    assert m.shape == (50, 50)
    assert 0.0 <= m.min() and m.max() <= 1.0
    assert m[25, 25] > 0.5 and m[2, 2] < 0.5


def test_points_in_masks_keeps_object_drops_background(tmp_path):
    p = tmp_path / "frame_0001.png"
    _write_center_mask(p, box=20)
    on_object = Point3D(1, np.array([0.0, 0.0, 0.0]), np.zeros(3), 0.0, np.array([7]), np.array([0]))
    off_object = Point3D(2, np.array([3.0, 3.0, 0.0]), np.zeros(3), 0.0, np.array([7]), np.array([0]))

    keep = _points_in_masks(
        {1: on_object, 2: off_object}, image_ids=[7], camtoworlds=_cam_at(),
        Ks=K[None], widths=[W], heights=[H], undistort_maps=[None], mask_paths=[p], dilate_px=0,
    )
    assert list(keep) == [True, False]


def test_points_in_masks_needs_min_views(tmp_path):
    p = tmp_path / "frame_0001.png"
    _write_center_mask(p, box=20)
    # observed only by an image with no mask -> can't be confirmed on-object
    pt = Point3D(1, np.array([0.0, 0.0, 0.0]), np.zeros(3), 0.0, np.array([9]), np.array([0]))
    keep = _points_in_masks(
        {1: pt}, image_ids=[7], camtoworlds=_cam_at(),
        Ks=K[None], widths=[W], heights=[H], undistort_maps=[None], mask_paths=[p],
    )
    assert list(keep) == [False]


def test_gaussians_in_masks(tmp_path):
    p = tmp_path / "frame_0001.png"
    _write_center_mask(p, box=25)
    scene = SceneData(
        image_paths=[], image_names=["frame_0001.jpg"], camtoworlds=_cam_at(),
        Ks=K[None], widths=np.array([W]), heights=np.array([H]), undistort_maps=[None],
        points_xyz=np.zeros((0, 3), np.float32), points_rgb=np.zeros((0, 3), np.uint8),
        scene_scale=1.0, transform=np.eye(4), mask_paths=[p],
    )
    means = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 0.0]])  # centre (in mask), far corner (out)
    keep = gaussians_in_masks(means, scene, np.array([0]), dilate_px=0)
    assert list(keep) == [True, False]


def test_gaussians_in_masks_no_masks_keeps_all():
    scene = SceneData(
        image_paths=[], image_names=[], camtoworlds=_cam_at(), Ks=K[None],
        widths=np.array([W]), heights=np.array([H]), undistort_maps=[None],
        points_xyz=np.zeros((0, 3), np.float32), points_rgb=np.zeros((0, 3), np.uint8),
        scene_scale=1.0, transform=np.eye(4), mask_paths=None,
    )
    keep = gaussians_in_masks(np.zeros((5, 3)), scene, np.array([0]))
    assert keep.all()
