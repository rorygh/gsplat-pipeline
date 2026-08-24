"""Fast, self-contained test for the COLMAP binary-format reader: writes a
tiny synthetic model with the same byte layout `colmap` itself produces, then
checks the reader round-trips it correctly. No dependency on COLMAP or GPU.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from gsplat_pipeline.colmap.binary import qvec2rotmat, read_model


def _write_cameras_bin(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))  # num_cameras
        # camera_id=1, model_id=1 (PINHOLE), width=100, height=80, 4 params
        f.write(struct.pack("<iiQQ", 1, 1, 100, 80))
        f.write(struct.pack("<dddd", 60.0, 60.0, 50.0, 40.0))  # fx, fy, cx, cy


def _write_images_bin(path: Path, names: list[str]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(names, start=1):
            # identity rotation, translation = (i, 0, 0), camera_id=1
            f.write(struct.pack("<idddddddi", i, 1.0, 0.0, 0.0, 0.0, float(i), 0.0, 0.0, 1))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D points


def _write_points3D_bin(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 2))  # num_points
        for pid in range(1, 3):
            f.write(struct.pack("<QdddBBBd", pid, float(pid), 0.0, 0.0, 255, 0, 0, 0.1))
            f.write(struct.pack("<Q", 0))  # empty track


@pytest.fixture
def sparse_model(tmp_path: Path) -> Path:
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    _write_cameras_bin(sparse_dir / "cameras.bin")
    _write_images_bin(sparse_dir / "images.bin", ["img_b.png", "img_a.png"])
    _write_points3D_bin(sparse_dir / "points3D.bin")
    return sparse_dir


def test_read_model_roundtrip(sparse_model: Path):
    cameras, images, points3D = read_model(sparse_model)

    assert len(cameras) == 1
    cam = cameras[1]
    assert cam.model == "PINHOLE"
    assert cam.width == 100 and cam.height == 80
    np.testing.assert_allclose(cam.params, [60.0, 60.0, 50.0, 40.0])

    assert len(images) == 2
    assert {im.name for im in images.values()} == {"img_a.png", "img_b.png"}
    im1 = images[1]
    np.testing.assert_allclose(qvec2rotmat(im1.qvec), np.eye(3))
    np.testing.assert_allclose(im1.tvec, [1.0, 0.0, 0.0])

    assert len(points3D) == 2
    assert points3D[1].rgb.tolist() == [255, 0, 0]


def test_camera_intrinsics(sparse_model: Path):
    cameras, _, _ = read_model(sparse_model)
    K, dist = cameras[1].as_intrinsics()
    np.testing.assert_allclose(K, [[60, 0, 50], [0, 60, 40], [0, 0, 1]])
    np.testing.assert_allclose(dist, [0, 0, 0, 0])
