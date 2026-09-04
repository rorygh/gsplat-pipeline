"""Fast, CPU-only tests for the orbit-frame alignment."""

from __future__ import annotations

import numpy as np

from gsplat_pipeline.orientation import (
    apply_to_camtoworlds,
    apply_to_points,
    orbit_frame,
)


def _tilted_orbit(n=24, radius=3.0, tilt_deg=35.0, yaw_deg=50.0, height=0.4):
    """A circular orbit in a plane tilted off horizontal, as camera-to-world
    matrices (OpenCV convention) all looking at the origin."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(n, height)], axis=1)

    tx = np.radians(tilt_deg)
    rot_x = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    ty = np.radians(yaw_deg)
    rot_z = np.array([[np.cos(ty), -np.sin(ty), 0], [np.sin(ty), np.cos(ty), 0], [0, 0, 1]])
    world_from_plane = rot_z @ rot_x

    centers = circle @ world_from_plane.T
    target = np.zeros(3)

    c2w = np.zeros((n, 4, 4))
    for i, c in enumerate(centers):
        fwd = target - c
        fwd /= np.linalg.norm(fwd)
        # world up hint for building a sane camera basis: the plane normal
        up = world_from_plane @ np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        c2w[i, :3, 0] = right
        c2w[i, :3, 1] = down
        c2w[i, :3, 2] = fwd
        c2w[i, :3, 3] = c
        c2w[i, 3, 3] = 1.0
    return c2w, world_from_plane


def test_orbit_frame_flattens_and_levels():
    c2w, _ = _tilted_orbit()
    t = orbit_frame(c2w[:, :3, 3], camera_down=c2w[:, :3, 1])
    aligned = apply_to_camtoworlds(t, c2w)
    centers = aligned[:, :3, 3]

    # orbit now lies in a horizontal (z ~ const) plane, centred on the origin
    assert np.allclose(centers[:, 2], centers[0, 2], atol=1e-6)
    assert np.linalg.norm(centers.mean(axis=0)) < 1e-6
    # radius preserved (rigid transform)
    assert np.allclose(np.linalg.norm(centers, axis=1), 3.0, atol=1e-6)


def test_orbit_frame_up_points_up_not_down():
    c2w, _ = _tilted_orbit(tilt_deg=20.0)
    t = orbit_frame(c2w[:, :3, 3], camera_down=c2w[:, :3, 1])
    aligned = apply_to_camtoworlds(t, c2w)
    # cameras look roughly inward+level, so their local "down" (+Y) should have
    # a downward (-Z) world component after alignment
    assert aligned[:, 2, 1].mean() < 0


def test_orbit_frame_first_camera_on_positive_x():
    c2w, _ = _tilted_orbit(yaw_deg=123.0)
    t = orbit_frame(c2w[:, :3, 3], camera_down=c2w[:, :3, 1])
    first = apply_to_camtoworlds(t, c2w)[0, :3, 3]
    assert first[0] > 0
    assert abs(first[1]) < 1e-6


def test_orbit_frame_deterministic():
    c2w, _ = _tilted_orbit()
    a = orbit_frame(c2w[:, :3, 3], camera_down=c2w[:, :3, 1])
    b = orbit_frame(c2w[:, :3, 3], camera_down=c2w[:, :3, 1])
    np.testing.assert_array_equal(a, b)


def test_orbit_frame_identity_when_too_few_cameras():
    np.testing.assert_array_equal(orbit_frame(np.zeros((2, 3))), np.eye(4))


def test_apply_to_points_matches_camtoworld_translation():
    c2w, _ = _tilted_orbit()
    t = orbit_frame(c2w[:, :3, 3], camera_down=c2w[:, :3, 1])
    pts = c2w[:, :3, 3].astype(np.float32)
    np.testing.assert_allclose(
        apply_to_points(t, pts), apply_to_camtoworlds(t, c2w)[:, :3, 3], atol=1e-4
    )


def test_alignment_preserves_pairwise_distances():
    c2w, _ = _tilted_orbit()
    t = orbit_frame(c2w[:, :3, 3], camera_down=c2w[:, :3, 1])
    pts = np.random.default_rng(0).normal(size=(50, 3))
    out = apply_to_points(t, pts)
    d_before = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d_after = np.linalg.norm(out[:, None] - out[None], axis=-1)
    np.testing.assert_allclose(d_before, d_after, atol=1e-9)
