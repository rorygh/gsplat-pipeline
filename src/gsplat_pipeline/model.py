"""Gaussian Splatting scene representation: parameter initialization from a
COLMAP point cloud, and the per-parameter Adam optimizers/schedulers used to
train it. Mirrors the initialization used by gsplat's own examples and by
nerfstudio's splatfacto model (same defaults), just without the surrounding
multi-method training framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

SH_C0 = 0.28209479177387814  # DC term normalization for real spherical harmonics


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * SH_C0 + 0.5


def num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2


def _knn_distances(points: torch.Tensor, k: int, chunk_size: int = 4096) -> torch.Tensor:
    """Mean distance to the `k` nearest neighbors of each point, computed by
    brute-force chunked `cdist` (no scikit-learn dependency). Fine up to a
    few hundred thousand points, which covers typical SfM point clouds."""
    n = points.shape[0]
    out = torch.empty(n, device=points.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(points[start:end], points)  # [chunk, n]
        knn_dists, _ = dists.topk(k + 1, dim=-1, largest=False)  # includes self at distance 0
        out[start:end] = knn_dists[:, 1:].mean(dim=-1)
    return out


@dataclass
class GaussianModelConfig:
    sh_degree: int = 3
    init_opacity: float = 0.1
    init_scale: float = 1.0
    """Multiplier applied to the nearest-neighbor distance used for initial gaussian scale."""
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20


def init_gaussians(
    points_xyz: torch.Tensor,
    points_rgb: torch.Tensor,
    config: GaussianModelConfig,
    device: str = "cuda",
) -> torch.nn.ParameterDict:
    """Build the learnable Gaussian parameters, seeded from SfM points.

    points_rgb is expected in [0, 1] float.
    """
    n = points_xyz.shape[0]
    k = min(4, n - 1) if n > 1 else 1
    dist = _knn_distances(points_xyz, k=k)
    scales = torch.log(dist * config.init_scale).unsqueeze(-1).repeat(1, 3)

    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0  # identity rotation
    opacities = torch.logit(torch.full((n,), config.init_opacity))

    dim_sh = num_sh_bases(config.sh_degree)
    colors = torch.zeros(n, dim_sh, 3)
    colors[:, 0, :] = rgb_to_sh(points_rgb)

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points_xyz),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(colors[:, :1, :]),
            "shN": torch.nn.Parameter(colors[:, 1:, :]),
        }
    )
    return params.to(device)


def build_optimizers(
    params: torch.nn.ParameterDict, config: GaussianModelConfig, scene_scale: float
) -> Dict[str, torch.optim.Optimizer]:
    lrs = {
        "means": config.means_lr * scene_scale,
        "scales": config.scales_lr,
        "quats": config.quats_lr,
        "opacities": config.opacities_lr,
        "sh0": config.sh0_lr,
        "shN": config.shN_lr,
    }
    return {name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15) for name, lr in lrs.items()}


def build_means_scheduler(optimizers: Dict[str, torch.optim.Optimizer], max_steps: int) -> torch.optim.lr_scheduler.ExponentialLR:
    """The means' learning rate decays exponentially to 1% of its initial value
    over training -- standard practice since the original 3DGS paper."""
    gamma = 0.01 ** (1.0 / max_steps)
    return torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=gamma)


def get_colors(params: torch.nn.ParameterDict) -> torch.Tensor:
    """Full per-gaussian SH coefficient tensor, [N, num_sh_bases, 3], ready for `rasterization(..., colors=...)`."""
    return torch.cat([params["sh0"], params["shN"]], dim=1)


def get_scales(params: torch.nn.ParameterDict) -> torch.Tensor:
    return torch.exp(params["scales"])


def get_opacities(params: torch.nn.ParameterDict) -> torch.Tensor:
    return torch.sigmoid(params["opacities"])
