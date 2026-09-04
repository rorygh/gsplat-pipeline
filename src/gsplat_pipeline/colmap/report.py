"""Sanity-check visualisations for a COLMAP sparse model.

Answers "did SfM come out healthy?" without needing to open a 3D viewer:
which images registered, how well-connected they are, the camera path, and
the reprojection-error distribution. Deliberately does *not* render the
sparse point cloud -- that needs a real 3D viewer to read, and the camera
geometry + per-image stats are what actually tell you whether training will
work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..orientation import apply_to_camtoworlds, orbit_frame
from .binary import qvec2rotmat, read_model


@dataclass
class ColmapReport:
    num_cameras: int
    num_registered_images: int
    num_points: int
    mean_track_length: float
    mean_reproj_error: float
    mean_obs_per_image: float
    camera_models: dict


def _camtoworlds(images) -> tuple[list[str], np.ndarray]:
    names, mats = [], []
    for im in sorted(images.values(), key=lambda i: i.name):
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(im.qvec)
        w2c[:3, 3] = im.tvec
        names.append(im.name)
        mats.append(np.linalg.inv(w2c))
    return names, np.stack(mats)


def build_report(sparse_dir: Path, image_dir: Optional[Path] = None, align: bool = True) -> tuple[ColmapReport, dict]:
    cameras, images, points3D = read_model(sparse_dir)

    names, c2w = _camtoworlds(images)

    points_xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float64).reshape(-1, 3)

    transform = np.eye(4)
    if align:
        # Same +Z-up orbit frame the trainer bakes into the scene, so the plots
        # match what the viewer shows.
        transform = orbit_frame(c2w[:, :3, 3], camera_down=c2w[:, :3, 1])
        c2w = apply_to_camtoworlds(transform, c2w)
        if len(points_xyz):
            points_xyz = points_xyz @ transform[:3, :3].T + transform[:3, 3]

    centers = c2w[:, :3, 3]
    forward = c2w[:, :3, 2]  # +Z is the view direction (OpenCV convention)

    track_lengths = np.array([len(p.image_ids) for p in points3D.values()])
    errors = np.array([p.error for p in points3D.values()])

    # observations per registered image: how many 3D points each image sees
    obs = {name: 0 for name in names}
    id_to_name = {im.id: im.name for im in images.values()}
    for p in points3D.values():
        for iid in p.image_ids:
            if iid in id_to_name:
                obs[id_to_name[iid]] += 1
    obs_counts = np.array([obs[n] for n in names])

    registered_disk_total = None
    if image_dir is not None and Path(image_dir).is_dir():
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        registered_disk_total = sum(1 for f in Path(image_dir).iterdir() if f.suffix in exts)

    model_counts: dict = {}
    for cam in cameras.values():
        model_counts[cam.model] = model_counts.get(cam.model, 0) + 1

    report = ColmapReport(
        num_cameras=len(cameras),
        num_registered_images=len(images),
        num_points=len(points3D),
        mean_track_length=float(track_lengths.mean()) if len(track_lengths) else 0.0,
        mean_reproj_error=float(errors.mean()) if len(errors) else 0.0,
        mean_obs_per_image=float(obs_counts.mean()) if len(obs_counts) else 0.0,
        camera_models=model_counts,
    )

    details = {
        "summary": report.__dict__,
        "aligned_to_orbit_frame": bool(align),
        "orbit_tilt_deg": float(np.degrees(np.arccos(np.clip(transform[2, 2], -1.0, 1.0)))),
        "images_on_disk": registered_disk_total,
        "unregistered_count": (
            registered_disk_total - len(images) if registered_disk_total is not None else None
        ),
        "per_image": [
            {"name": n, "observations": int(o), "center": c.tolist()}
            for n, o, c in zip(names, obs_counts, centers)
        ],
        "_arrays": {  # kept for the plotting pass, dropped before JSON dump
            "names": names,
            "centers": centers,
            "forward": forward,
            "points": points_xyz,
            "obs_counts": obs_counts,
            "track_lengths": track_lengths,
            "errors": errors,
        },
    }
    return report, details


def _plot(details: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = details["_arrays"]
    centers, forward, obs_counts = a["centers"], a["forward"], a["obs_counts"]
    errors, track_lengths = a["errors"], a["track_lengths"]
    aligned = details.get("aligned_to_orbit_frame", False)
    written = []

    # SfM points as faint context. Keep only those near the orbit (background
    # points can sit hundreds of units away) and subsample for a light figure.
    points = a.get("points")
    if points is not None and len(points):
        c_mid = centers.mean(axis=0)
        reach = 2.5 * np.linalg.norm(centers - c_mid, axis=1).max()
        points = points[np.linalg.norm(points - c_mid, axis=1) <= reach]
        if len(points) > 4000:
            points = points[np.random.default_rng(0).choice(len(points), 4000, replace=False)]

    # common plot bounds: the orbit plus a margin (points beyond just clip)
    margin = 1.4 * np.abs(centers - centers.mean(axis=0)).max()
    bounds = np.stack([centers.mean(axis=0) - margin, centers.mean(axis=0) + margin])

    # 1. camera path -- with +Z up: top-down is X-Y, elevation is X-Z
    up_note = " (+Z up, orbit in X-Y)" if aligned else ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (i, j, title) in zip(axes, [(0, 1, "top-down (X-Y)"), (0, 2, "elevation (X-Z)")]):
        if points is not None and len(points):
            ax.scatter(points[:, i], points[:, j], s=1, c="0.8", alpha=0.5, zorder=0, linewidths=0)
        ax.plot(centers[:, i], centers[:, j], "-", color="0.7", lw=1, zorder=1)
        sc = ax.scatter(centers[:, i], centers[:, j], c=np.arange(len(centers)),
                        cmap="viridis", s=40, zorder=3)
        scale = 0.15 * np.ptp(centers, axis=0).max()
        ax.quiver(centers[:, i], centers[:, j], forward[:, i], forward[:, j],
                  color="crimson", angles="xy", scale_units="xy",
                  scale=1.0 / scale, width=0.003, zorder=2)
        ax.set_title(f"camera path, {title}")
        ax.set_xlabel("XYZ"[i]); ax.set_ylabel("XYZ"[j])
        ax.set_xlim(bounds[0, i], bounds[1, i])
        ax.set_ylim(bounds[0, j], bounds[1, j])
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.3)
    fig.suptitle(f"camera path{up_note}")
    fig.colorbar(sc, ax=axes, label="image index (capture order)", shrink=0.8)
    p = out_dir / "cameras_path.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 2. 3D camera positions + orientations
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if points is not None and len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="0.8", alpha=0.4, linewidths=0)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="0.7", lw=1)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c=np.arange(len(centers)), cmap="viridis", s=30)
    scale = 0.10 * (bounds[1] - bounds[0]).max()
    ax.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
              forward[:, 0], forward[:, 1], forward[:, 2],
              length=scale, color="crimson", normalize=True)
    ax.set_title("camera poses (line = capture order, red = view direction)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)" if aligned else "Z")
    ax.set_xlim(bounds[:, 0]); ax.set_ylim(bounds[:, 1]); ax.set_zlim(bounds[:, 2])
    ax.set_box_aspect((1, 1, 1))
    if aligned:
        ax.view_init(elev=18, azim=-70)
    p = out_dir / "cameras_3d.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 3. health histograms: observations/image, track length, reproj error
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(np.arange(len(obs_counts)), obs_counts, color="steelblue")
    axes[0].set(title="3D points observed per image", xlabel="image index", ylabel="count")
    axes[0].axhline(obs_counts.mean(), color="crimson", ls="--", lw=1)
    axes[1].hist(track_lengths, bins=30, color="steelblue")
    axes[1].set(title="track length (images per 3D point)", xlabel="images", ylabel="points")
    axes[1].axvline(track_lengths.mean(), color="crimson", ls="--", lw=1)
    axes[2].hist(errors, bins=40, color="steelblue")
    axes[2].set(title="reprojection error", xlabel="pixels", ylabel="points")
    axes[2].axvline(errors.mean(), color="crimson", ls="--", lw=1)
    fig.tight_layout()
    p = out_dir / "sfm_health.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    return written


def write_report(sparse_dir: Path, out_dir: Path, image_dir: Optional[Path] = None, align: bool = True) -> ColmapReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, details = build_report(Path(sparse_dir), image_dir, align=align)

    plotted = []
    try:
        plotted = _plot(details, out_dir)
    except ImportError:
        print("[colmap-report] matplotlib not installed -- writing JSON only "
              "(`pip install matplotlib` or `pip install gsplat-pipeline[viz]`)")

    details.pop("_arrays", None)
    with open(out_dir / "colmap_report.json", "w") as f:
        json.dump(details, f, indent=2)

    s = report
    if align:
        print(f"[colmap-report] aligned to +Z-up orbit frame "
              f"({details['orbit_tilt_deg']:.1f}deg tilt from COLMAP world)")
    unreg = details.get("unregistered_count")
    print(f"[colmap-report] {s.num_registered_images} images registered"
          + (f" ({unreg} not registered)" if unreg else "")
          + f", {s.num_cameras} camera(s) {s.camera_models}")
    print(f"[colmap-report] {s.num_points} points, mean track {s.mean_track_length:.1f} imgs, "
          f"mean reproj err {s.mean_reproj_error:.2f}px, mean {s.mean_obs_per_image:.0f} obs/image")
    for p in plotted:
        print(f"[colmap-report] wrote {p}")
    return report
