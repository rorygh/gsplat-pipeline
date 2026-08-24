"""Fast, CPU-only tests for Gaussian parameter initialization (no GPU/COLMAP needed)."""

from __future__ import annotations

import torch

from gsplat_pipeline.model import (
    GaussianModelConfig,
    build_means_scheduler,
    build_optimizers,
    get_colors,
    get_opacities,
    get_scales,
    init_gaussians,
    num_sh_bases,
    rgb_to_sh,
    sh_to_rgb,
)


def test_rgb_sh_roundtrip():
    rgb = torch.rand(100, 3)
    torch.testing.assert_close(sh_to_rgb(rgb_to_sh(rgb)), rgb)


def test_num_sh_bases():
    assert num_sh_bases(0) == 1
    assert num_sh_bases(3) == 16


def test_init_gaussians_shapes():
    n = 50
    points = torch.rand(n, 3)
    colors = torch.rand(n, 3)
    config = GaussianModelConfig(sh_degree=2)
    params = init_gaussians(points, colors, config, device="cpu")

    assert params["means"].shape == (n, 3)
    assert params["scales"].shape == (n, 3)
    assert params["quats"].shape == (n, 4)
    assert params["opacities"].shape == (n,)
    assert params["sh0"].shape == (n, 1, 3)
    assert params["shN"].shape == (n, num_sh_bases(2) - 1, 3)

    torch.testing.assert_close(params["means"], points)
    # identity rotation
    torch.testing.assert_close(params["quats"], torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1))

    colors_out = get_colors(params)
    assert colors_out.shape == (n, num_sh_bases(2), 3)
    assert torch.all(get_opacities(params) > 0) and torch.all(get_opacities(params) < 1)
    assert torch.all(get_scales(params) > 0)


def test_optimizers_and_scheduler():
    n = 20
    points = torch.rand(n, 3)
    colors = torch.rand(n, 3)
    config = GaussianModelConfig()
    params = init_gaussians(points, colors, config, device="cpu")
    optimizers = build_optimizers(params, config, scene_scale=2.0)

    assert set(optimizers.keys()) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    assert optimizers["means"].param_groups[0]["lr"] == config.means_lr * 2.0

    scheduler = build_means_scheduler(optimizers, max_steps=1000)
    initial_lr = optimizers["means"].param_groups[0]["lr"]
    for _ in range(1000):
        scheduler.step()
    final_lr = optimizers["means"].param_groups[0]["lr"]
    torch.testing.assert_close(final_lr / initial_lr, 0.01, atol=1e-6, rtol=1e-3)
