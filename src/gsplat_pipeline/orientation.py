"""Recover a natural world frame from the camera trajectory.

COLMAP fixes the world frame arbitrarily (gauge freedom -- the reconstruction
is only defined up to a rigid transform + scale). For an object-centric
capture where the camera orbits roughly in a plane, there *is* a natural
frame: **+Z up**, normal to the orbit plane, with the orbit centre at the
origin and the capture starting on the +X axis.

`orbit_frame` estimates the rigid transform to that frame from the camera
centres (plus a hint from where "up" is in each camera). It's applied to the
COLMAP report plots and baked into the trained checkpoint / exported PLY, so
every downstream viewer opens the scene the same way up.

Assumption: the orbit is roughly planar. If it isn't (a free-flight capture,
a sphere of views) the plane fit is meaningless -- pass `align=False`.
"""

from __future__ import annotations

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = _skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector `a` onto unit vector `b`."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-8:  # antiparallel: 180 deg about any perpendicular axis
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return _rot_about(perp, np.pi)
    vx = _skew(v)
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _circle_center_2d(xy: np.ndarray) -> np.ndarray:
    """Algebraic (Kåsa) circle fit -- returns the centre, or the centroid if
    the points are too collinear for the fit to mean anything."""
    x, y = xy[:, 0], xy[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    except np.linalg.LinAlgError:
        return xy.mean(axis=0)
    center = np.array([sol[0] / 2.0, sol[1] / 2.0])
    spread = np.linalg.norm(xy - xy.mean(axis=0), axis=1).max()
    if not np.all(np.isfinite(center)) or np.linalg.norm(center - xy.mean(axis=0)) > 10 * spread:
        return xy.mean(axis=0)
    return center


def _translate(t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def orbit_frame(camera_centers: np.ndarray, camera_down: np.ndarray | None = None) -> np.ndarray:
    """Rigid 4x4 transform ``T`` mapping COLMAP world coords to an orbit frame
    (``p_new = T @ [p; 1]``): +Z is the orbit-plane normal pointing "up", the
    origin sits at the orbit centre (circle fit through the camera path) on the
    camera plane, and the first camera lies on the +X axis.

    camera_centers: (N, 3) camera positions in world space.
    camera_down:    (N, 3) each camera's local +Y (down, OpenCV convention) in
                    world space, used only to disambiguate which way is up.
                    If omitted, the COLMAP +Z axis is used as the up hint.
    """
    centers = np.asarray(camera_centers, dtype=np.float64)
    if centers.shape[0] < 3:
        return np.eye(4)

    centroid = centers.mean(axis=0)

    # best-fit plane normal = direction of least variance about the centroid
    _, _, vt = np.linalg.svd(centers - centroid, full_matrices=False)
    normal = vt[-1]

    if camera_down is not None and len(camera_down):
        up_hint = -np.asarray(camera_down, dtype=np.float64).mean(axis=0)
    else:
        up_hint = np.array([0.0, 0.0, 1.0])
    if np.dot(normal, up_hint) < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)

    r_up = np.eye(4)
    r_up[:3, :3] = _shortest_arc(normal, np.array([0.0, 0.0, 1.0]))

    # in the levelled frame, move the origin to the orbit centre on the camera plane
    levelled = r_up @ _translate(-centroid)
    planar = centers @ levelled[:3, :3].T + levelled[:3, 3]
    cx, cy = _circle_center_2d(planar[:, :2])
    recentre = _translate([-cx, -cy, -planar[:, 2].mean()])

    t = recentre @ levelled

    # deterministic in-plane yaw: rotate so the first camera sits on +X
    first = (t @ np.r_[centers[0], 1.0])[:3]
    yaw = np.arctan2(first[1], first[0])
    r_yaw = np.eye(4)
    r_yaw[:3, :3] = _rot_about(np.array([0.0, 0.0, 1.0]), -yaw)
    return r_yaw @ t


def apply_to_camtoworlds(transform: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    """Left-multiply a stack of (N, 4, 4) camera-to-world matrices by `transform`."""
    return np.einsum("ij,njk->nik", transform, camtoworlds)


def apply_to_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """Transform an (M, 3) point cloud by the homogeneous `transform`."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    out = pts @ transform[:3, :3].T + transform[:3, 3]
    return out.astype(points_xyz.dtype, copy=False)
