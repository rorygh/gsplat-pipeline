"""Reader for COLMAP's sparse-model binary format.

COLMAP writes its sparse reconstruction (`sparse/0/{cameras,images,points3D}.bin`)
in a small, stable binary format documented at https://colmap.github.io/format.html.
This module reads it directly with `struct`, with no dependency on COLMAP itself
or on `pycolmap` (whose wheel bundles its own COLMAP build, which can silently
diverge from whatever `colmap` binary actually produced the model).

Only the camera models COLMAP's default feature/mapper pipeline actually
produces are supported: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, and
OPENCV. That covers every model `ns-process-data` / plain `colmap
automatic_reconstructor` output.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

# (num_params, has_radial, has_tangential) is not needed; we only need the
# raw param count and a name -> id map matching COLMAP's src/colmap/sensor/models.h
CAMERA_MODEL_NUM_PARAMS = {
    0: 3,  # SIMPLE_PINHOLE: f, cx, cy
    1: 4,  # PINHOLE: fx, fy, cx, cy
    2: 4,  # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,  # RADIAL: f, cx, cy, k1, k2
    4: 8,  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
}
CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # meaning depends on `model`, see CAMERA_MODEL_NUM_PARAMS

    def as_intrinsics(self) -> "tuple[np.ndarray, np.ndarray]":
        """Return (K, dist) as a pinhole calibration matrix + OpenCV-style [k1,k2,p1,p2] distortion."""
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
            dist = np.zeros(4, dtype=np.float64)
        elif self.model == "PINHOLE":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            dist = np.zeros(4, dtype=np.float64)
        elif self.model == "SIMPLE_RADIAL":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
            dist = np.array([p[3], 0.0, 0.0, 0.0])
        elif self.model == "RADIAL":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
            dist = np.array([p[3], p[4], 0.0, 0.0])
        elif self.model == "OPENCV":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            dist = np.array([p[4], p[5], p[6], p[7]])
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {self.model}")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return K, dist


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (4,) w,x,y,z quaternion, world-to-camera rotation
    tvec: np.ndarray  # (3,) world-to-camera translation
    camera_id: int
    name: str
    point3D_ids: np.ndarray  # (num_points2D,) with -1 for unmatched keypoints


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray  # (3,)
    rgb: np.ndarray  # (3,) uint8
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternions are scalar-first (w, x, y, z)."""
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )


def _read(fid, fmt: str, endian: str = "<"):
    size = struct.calcsize(endian + fmt)
    data = fid.read(size)
    return struct.unpack(endian + fmt, data)


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with open(path, "rb") as fid:
        (num_cameras,) = _read(fid, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read(fid, "iiQQ")
            num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.array(_read(fid, "d" * num_params))
            cameras[camera_id] = Camera(
                id=camera_id,
                model=CAMERA_MODEL_NAMES[model_id],
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_images_binary(path: Path) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, "rb") as fid:
        (num_images,) = _read(fid, "Q")
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(fid, "idddddddi")
            name = b""
            while True:
                (c,) = _read(fid, "c")
                if c == b"\x00":
                    break
                name += c
            (num_points2D,) = _read(fid, "Q")
            xys_and_ids = _read(fid, "ddq" * num_points2D)
            point3D_ids = np.array(xys_and_ids[2::3], dtype=np.int64)
            images[image_id] = Image(
                id=image_id,
                qvec=np.array([qw, qx, qy, qz]),
                tvec=np.array([tx, ty, tz]),
                camera_id=camera_id,
                name=name.decode("utf-8"),
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_binary(path: Path) -> Dict[int, Point3D]:
    points: Dict[int, Point3D] = {}
    with open(path, "rb") as fid:
        (num_points,) = _read(fid, "Q")
        for _ in range(num_points):
            point3D_id, x, y, z, r, g, b, error = _read(fid, "QdddBBBd")
            (track_length,) = _read(fid, "Q")
            track = _read(fid, "ii" * track_length)
            image_ids = np.array(track[0::2], dtype=np.int64)
            point2D_idxs = np.array(track[1::2], dtype=np.int64)
            points[point3D_id] = Point3D(
                id=point3D_id,
                xyz=np.array([x, y, z]),
                rgb=np.array([r, g, b], dtype=np.uint8),
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points


def read_model(sparse_dir: Path):
    """Read a full COLMAP sparse model directory. Returns (cameras, images, points3D)."""
    sparse_dir = Path(sparse_dir)
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    points3D = read_points3D_binary(sparse_dir / "points3D.bin")
    return cameras, images, points3D
