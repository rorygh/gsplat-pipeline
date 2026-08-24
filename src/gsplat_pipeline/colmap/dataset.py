"""Turn a COLMAP sparse model + images folder into cameras, poses, and a
training/eval split ready for the Gaussian Splatting trainer.

Cameras are undistorted once up front (like gsplat's own example loader) so
the rasterizer only ever has to deal with ideal pinhole projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from .binary import qvec2rotmat, read_model

MAX_AUTO_RESOLUTION = 1600
"""Matches nerfstudio's ColmapDataParser: auto-picked downscale keeps the
long edge under this many pixels, so full-resolution photos (common straight
out of a camera) don't make training absurdly slow by default."""


@dataclass
class SceneData:
    image_paths: List[Path]
    image_names: List[str]
    camtoworlds: np.ndarray  # (N, 4, 4) float64, OpenCV convention (+X right, +Y down, +Z forward)
    Ks: np.ndarray  # (N, 3, 3) undistorted intrinsics, one per image
    widths: np.ndarray  # (N,) int, post-downscale/undistortion
    heights: np.ndarray  # (N,) int, post-downscale/undistortion
    undistort_maps: List[Optional[tuple]]  # per-image (mapx, mapy) or None if already pinhole
    points_xyz: np.ndarray  # (M, 3) float32, sparse SfM points
    points_rgb: np.ndarray  # (M, 3) uint8
    scene_scale: float  # rough scene radius, used to scale learning rates / thresholds


def _undistort_image(image: np.ndarray, mapx, mapy) -> np.ndarray:
    if mapx is None:
        return image
    return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)


def _pick_auto_downscale_factor(width: int, height: int) -> int:
    factor = 1
    while max(width, height) / factor > MAX_AUTO_RESOLUTION:
        factor *= 2
    return factor


def load_scene(
    data_dir: Path,
    sparse_path: Optional[Path] = None,
    images_path: Optional[Path] = None,
    downscale_factor: Union[int, str, None] = "auto",
) -> SceneData:
    """Load a COLMAP reconstruction. `data_dir` is expected to contain an
    `images/` folder and a `sparse/0/` (or `colmap/sparse/0/`) model, unless
    overridden via `sparse_path` / `images_path`.

    `downscale_factor`: an explicit power-of-2 factor, `"auto"` (mirrors
    nerfstudio: picks the smallest factor keeping the long edge under
    `MAX_AUTO_RESOLUTION`, falling back to full resolution if a matching
    `images_{factor}/` folder isn't present -- as shipped by e.g. the
    Mip-NeRF 360 dataset), or `None`/`1` for full resolution. Only pre-existing
    `images_{factor}/` folders are used; this pipeline doesn't generate them.
    """
    data_dir = Path(data_dir)
    if sparse_path is None:
        for candidate in ["sparse/0", "colmap/sparse/0", "sparse"]:
            if (data_dir / candidate / "cameras.bin").exists():
                sparse_path = data_dir / candidate
                break
        else:
            raise FileNotFoundError(f"No COLMAP sparse model found under {data_dir} (checked sparse/0, colmap/sparse/0)")

    cameras, images, points3D = read_model(sparse_path)

    first_cam = next(iter(cameras.values()))
    if downscale_factor == "auto":
        factor = _pick_auto_downscale_factor(first_cam.width, first_cam.height)
        if factor > 1 and images_path is None and not (data_dir / f"images_{factor}").exists():
            print(f"[data] would downscale {factor}x, but {data_dir / f'images_{factor}'} doesn't exist -- using full resolution")
            factor = 1
    else:
        factor = downscale_factor or 1

    if images_path is None:
        images_path = data_dir / (f"images_{factor}" if factor > 1 else "images")
    if factor > 1:
        print(f"[data] using {factor}x downscaled images from {images_path}")

    image_ids = sorted(images.keys(), key=lambda i: images[i].name)
    camtoworlds = []
    Ks = []
    widths = []
    heights = []
    undistort_maps: List[Optional[tuple]] = []
    image_paths = []
    image_names = []

    for image_id in image_ids:
        im = images[image_id]
        cam = cameras[im.camera_id]
        K, dist = cam.as_intrinsics()
        target_w, target_h = cam.width // factor, cam.height // factor
        if factor > 1:
            K = K.copy()
            K[:2, :] /= factor

        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(im.qvec)
        w2c[:3, 3] = im.tvec
        camtoworlds.append(np.linalg.inv(w2c))

        if np.any(dist != 0):
            new_K, _roi = cv2.getOptimalNewCameraMatrix(K, dist, (target_w, target_h), alpha=0)
            mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (target_w, target_h), cv2.CV_32FC1)
            undistort_maps.append((mapx, mapy))
            Ks.append(new_K)
        else:
            undistort_maps.append(None)
            Ks.append(K)
        widths.append(target_w)
        heights.append(target_h)

        image_paths.append(images_path / im.name)
        image_names.append(im.name)

    camtoworlds = np.stack(camtoworlds, axis=0)
    Ks = np.stack(Ks, axis=0)
    widths = np.array(widths)
    heights = np.array(heights)

    points_xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float32)
    points_rgb = np.array([p.rgb for p in points3D.values()], dtype=np.uint8)

    camera_locations = camtoworlds[:, :3, 3]
    scene_center = camera_locations.mean(axis=0)
    scene_scale = float(np.linalg.norm(camera_locations - scene_center, axis=1).max())

    return SceneData(
        image_paths=image_paths,
        image_names=image_names,
        camtoworlds=camtoworlds,
        Ks=Ks,
        widths=widths,
        heights=heights,
        undistort_maps=undistort_maps,
        points_xyz=points_xyz,
        points_rgb=points_rgb,
        scene_scale=scene_scale,
    )


def train_eval_split(num_images: int, eval_every: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Every `eval_every`-th image (by sorted filename order) is held out for eval.
    Matches the convention used by nerfstudio's ColmapDataParser and gsplat's own
    examples, so PSNR numbers are comparable to published Mip-NeRF 360 results."""
    indices = np.arange(num_images)
    eval_idx = indices[indices % eval_every == 0]
    train_idx = indices[indices % eval_every != 0]
    return train_idx, eval_idx


class GaussianSplattingDataset(torch.utils.data.Dataset):
    """Lazily loads + undistorts one (image, pose, intrinsics) pair per item."""

    def __init__(self, scene: SceneData, indices: np.ndarray):
        self.scene = scene
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = self.indices[item]
        scene = self.scene
        image = cv2.cvtColor(cv2.imread(str(scene.image_paths[idx])), cv2.COLOR_BGR2RGB)
        mapx, mapy = scene.undistort_maps[idx] or (None, None)
        image = _undistort_image(image, mapx, mapy)
        return {
            "image": torch.from_numpy(image).float(),
            "K": torch.from_numpy(scene.Ks[idx]).float(),
            "camtoworld": torch.from_numpy(scene.camtoworlds[idx]).float(),
            "width": int(scene.widths[idx]),
            "height": int(scene.heights[idx]),
            "image_name": scene.image_names[idx],
            "image_id": int(idx),
        }
