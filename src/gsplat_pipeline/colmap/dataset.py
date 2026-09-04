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

from ..orientation import apply_to_camtoworlds, apply_to_points, orbit_frame
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
    transform: np.ndarray  # (4, 4) rigid map from raw COLMAP world coords to the frame above (identity if unaligned)
    mask_paths: Optional[List[Optional[Path]]] = None  # per-image object mask, or None (no masking / missing)


def _undistort_image(image: np.ndarray, mapx, mapy) -> np.ndarray:
    if mapx is None:
        return image
    return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)


def _resolve_mask_paths(mask_dir: Optional[Path], image_names: List[str]) -> Optional[List[Optional[Path]]]:
    """Match `mask_dir/<stem>.png` to each image. Returns None if no mask dir;
    a per-image list (with None for any image missing a mask) otherwise."""
    if mask_dir is None:
        return None
    mask_dir = Path(mask_dir)
    paths: List[Optional[Path]] = []
    for name in image_names:
        p = mask_dir / f"{Path(name).stem}.png"
        paths.append(p if p.exists() else None)
    found = sum(p is not None for p in paths)
    if found == 0:
        raise FileNotFoundError(f"--mask-dir {mask_dir} has no <stem>.png matching the {len(image_names)} images")
    if found < len(image_names):
        print(f"[data] mask: {found}/{len(image_names)} images have a mask (rest train on the full frame)")
    return paths


def _load_working_mask(path: Optional[Path], w: int, h: int, undistort_map, nearest: bool = False) -> Optional[np.ndarray]:
    """Load a full-res mask PNG and bring it into the working (downscaled +
    undistorted) frame the poses/intrinsics live in. Returns float 0-1 (H, W)."""
    if path is None:
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    if undistort_map is not None:
        m = cv2.remap(m, undistort_map[0], undistort_map[1],
                      cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    return m.astype(np.float32) / 255.0


def _points_in_masks(
    points3D, image_ids, camtoworlds, Ks, widths, heights, undistort_maps,
    mask_paths, min_views: int = 1, dilate_px: int = 9,
) -> np.ndarray:
    """Boolean keep-mask over `points3D.values()`: True where the point projects
    inside the (dilated) object mask in >= min_views of its observing images.
    Everything the object mask excludes -- the entire background -- is dropped,
    which is what we want for an object-only splat (and mandatory once the
    object moves, since background points are geometrically inconsistent)."""
    id_to_idx = {int(cid): i for i, cid in enumerate(image_ids)}
    cache: dict = {}

    def masky(i: int):
        if i not in cache:
            m = _load_working_mask(mask_paths[i], int(widths[i]), int(heights[i]), undistort_maps[i], nearest=True)
            if m is not None and dilate_px:
                m = cv2.dilate((m > 0.5).astype(np.uint8), np.ones((dilate_px, dilate_px), np.uint8))
                m = m.astype(bool)
            elif m is not None:
                m = m > 0.5
            cache[i] = m
        return cache[i]

    plist = list(points3D.values())
    keep = np.zeros(len(plist), dtype=bool)
    for pi, p in enumerate(plist):
        xyz = np.asarray(p.xyz, dtype=np.float64)
        hits = 0
        for cid in p.image_ids:
            i = id_to_idx.get(int(cid))
            if i is None:
                continue
            m = masky(i)
            if m is None:
                continue
            c2w = camtoworlds[i]
            rot = c2w[:3, :3].T
            xc = rot @ (xyz - c2w[:3, 3])
            if xc[2] <= 1e-6:
                continue
            uv = Ks[i] @ xc
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
            if 0 <= v < m.shape[0] and 0 <= u < m.shape[1] and m[v, u]:
                hits += 1
                if hits >= min_views:
                    break
        keep[pi] = hits >= min_views
    return keep


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
    align: bool = True,
    mask_dir: Optional[Path] = None,
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

    # Object masks: resolve per-image, and drop every SfM point that isn't on
    # the object so no Gaussian is seeded in the background.
    mask_paths = _resolve_mask_paths(mask_dir, image_names)
    if mask_paths is not None and len(points_xyz):
        keep = _points_in_masks(points3D, image_ids, camtoworlds, Ks, widths, heights, undistort_maps, mask_paths)
        print(f"[data] mask: kept {int(keep.sum())}/{len(keep)} SfM points on the object")
        points_xyz, points_rgb = points_xyz[keep], points_rgb[keep]

    # Recover a +Z-up orbit frame from the camera trajectory and bake it in, so
    # the trained scene (checkpoint + PLY) and every viewer share one orientation.
    transform = np.eye(4)
    if align:
        transform = orbit_frame(camtoworlds[:, :3, 3], camera_down=camtoworlds[:, :3, 1])
        camtoworlds = apply_to_camtoworlds(transform, camtoworlds)
        points_xyz = apply_to_points(transform, points_xyz)
        tilt = np.degrees(np.arccos(np.clip(transform[2, 2], -1.0, 1.0)))
        print(f"[data] aligned to orbit frame (+Z up), rotated {tilt:.1f}deg from COLMAP world")

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
        transform=transform,
        mask_paths=mask_paths,
    )


def gaussians_in_masks(
    means: np.ndarray, scene: "SceneData", indices: np.ndarray, dilate_px: int = 15,
) -> np.ndarray:
    """Boolean keep-mask over `means` (N, 3): True where the centre projects into
    the object mask of at least one of the given views. Used to prune a trained
    model down to the object."""
    keep = np.zeros(len(means), dtype=bool)
    if scene.mask_paths is None:
        return ~keep  # no masks -> keep everything
    m64 = np.asarray(means, dtype=np.float64)
    for i in indices:
        m = _load_working_mask(scene.mask_paths[i], int(scene.widths[i]), int(scene.heights[i]),
                               scene.undistort_maps[i], nearest=True)
        if m is None:
            continue
        m = cv2.dilate((m > 0.5).astype(np.uint8), np.ones((dilate_px, dilate_px), np.uint8)) > 0
        c2w = scene.camtoworlds[i]
        xc = (m64 - c2w[:3, 3]) @ c2w[:3, :3]  # world -> camera (rot^T applied on the right)
        front = xc[:, 2] > 1e-6
        uv = xc @ scene.Ks[i].T
        u = np.round(uv[:, 0] / np.where(front, uv[:, 2], 1.0)).astype(int)
        v = np.round(uv[:, 1] / np.where(front, uv[:, 2], 1.0)).astype(int)
        inb = front & (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])
        hit = np.zeros(len(means), dtype=bool)
        hit[inb] = m[v[inb], u[inb]]
        keep |= hit
    return keep


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
        h, w = image.shape[:2]

        mask = np.ones((h, w, 1), dtype=np.float32)
        if scene.mask_paths is not None:
            m = _load_working_mask(scene.mask_paths[idx], w, h, scene.undistort_maps[idx])
            if m is not None:
                mask = m[..., None]

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
            "K": torch.from_numpy(scene.Ks[idx]).float(),
            "camtoworld": torch.from_numpy(scene.camtoworlds[idx]).float(),
            "width": int(scene.widths[idx]),
            "height": int(scene.heights[idx]),
            "image_name": scene.image_names[idx],
            "image_id": int(idx),
        }
