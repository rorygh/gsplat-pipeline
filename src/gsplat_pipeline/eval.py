"""Evaluate a trained Gaussian scene on its held-out views: PSNR/SSIM plus
the rendered images themselves, so quality can be judged by eye and not just
by the metric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import imageio
import numpy as np
import torch
from gsplat.rendering import rasterization
from pytorch_msssim import ssim as ssim_fn
from tqdm import tqdm

from .colmap.dataset import GaussianSplattingDataset, load_scene, train_eval_split
from .io import load_checkpoint
from .model import get_colors, get_opacities, get_scales
from .train import _viewmat_from_camtoworld


@dataclass
class EvalConfig:
    data_dir: Path
    checkpoint: Path
    output_dir: Path
    colmap_path: Optional[Path] = None
    downscale_factor: Union[int, str, None] = "auto"
    align: bool = True
    mask_dir: Optional[Path] = None
    """If set, metrics are computed on the masked (object-only) region -- match
    this to how the checkpoint was trained, or the numbers penalise a model that
    was deliberately not asked to reconstruct the background."""
    eval_every: int = 8
    device: str = "cuda"


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - gt) ** 2)
    return -10.0 * torch.log10(mse)


@torch.no_grad()
def evaluate(cfg: EvalConfig) -> dict:
    device = cfg.device
    scene = load_scene(cfg.data_dir, sparse_path=cfg.colmap_path, downscale_factor=cfg.downscale_factor,
                       align=cfg.align, mask_dir=cfg.mask_dir)
    masked = scene.mask_paths is not None
    _, eval_idx = train_eval_split(len(scene.image_names), cfg.eval_every)
    eval_dataset = GaussianSplattingDataset(scene, eval_idx)

    params, meta = load_checkpoint(cfg.checkpoint, device=device)
    sh_degree = meta.get("sh_degree", 3)

    render_dir = Path(cfg.output_dir) / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    psnrs, ssims = [], []
    for i in tqdm(range(len(eval_dataset)), desc="evaluating"):
        item = eval_dataset[i]
        camtoworld = item["camtoworld"].to(device)[None]
        K = item["K"].to(device)[None]
        width, height = item["width"], item["height"]
        gt_image = (item["image"].to(device) / 255.0)[None]

        viewmat = _viewmat_from_camtoworld(camtoworld[0])[None]
        render, alpha, _ = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=get_scales(params),
            opacities=get_opacities(params),
            colors=get_colors(params),
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree,
        )
        background = torch.zeros(3, device=device)
        pred_image = torch.clamp(render[..., :3] + (1 - alpha) * background, 0.0, 1.0)

        if masked:
            m = item["mask"].to(device)[None]
            pred_m = pred_image * m
            gt_m = gt_image * m
        else:
            pred_m, gt_m = pred_image, gt_image

        psnrs.append(psnr(pred_m, gt_m).item())
        ssims.append(
            ssim_fn(pred_m.permute(0, 3, 1, 2), gt_m.permute(0, 3, 1, 2), data_range=1.0).item()
        )

        combined = torch.cat([gt_m[0] if masked else gt_image[0], pred_m[0] if masked else pred_image[0]], dim=1).clamp(0, 1)
        out = (combined.cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(render_dir / f"{item['image_name']}", out)

    results = {
        "num_images": len(eval_dataset),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "masked": bool(masked),
    }
    with open(Path(cfg.output_dir) / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    tag = " (masked region)" if masked else ""
    print(f"[eval] PSNR: {results['psnr']:.2f}  SSIM: {results['ssim']:.4f}  ({results['num_images']} held-out views){tag}")
    return results
