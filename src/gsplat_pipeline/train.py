"""Train a Gaussian Splatting scene from a COLMAP reconstruction.

The whole training loop -- render, loss, backward, densify -- fits in one
function. Densification (splitting/duplicating/pruning gaussians) is not
reimplemented here: it's delegated to `gsplat.strategy.DefaultStrategy`,
the same strategy nerfstudio's splatfacto model uses, which ships as part
of the `gsplat` library itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from pytorch_msssim import SSIM
from tqdm import tqdm

from .colmap.dataset import GaussianSplattingDataset, load_scene, train_eval_split
from .io import export_ply, save_checkpoint
from .model import (
    GaussianModelConfig,
    build_means_scheduler,
    build_optimizers,
    get_colors,
    get_opacities,
    get_scales,
    init_gaussians,
)


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    colmap_path: Optional[Path] = None
    """Explicit path to the COLMAP sparse model. Auto-detected under data_dir if unset."""
    downscale_factor: Union[int, str, None] = "auto"
    """Image downscale factor: an explicit power of 2, "auto" (nerfstudio-style: keeps
    the long edge under 1600px, using a pre-existing images_{factor}/ folder if present),
    or None/1 for full resolution."""

    max_steps: int = 30_000
    eval_every: int = 8
    """Every Nth image (by filename) is held out for eval, matching Mip-NeRF 360 convention."""

    ssim_lambda: float = 0.2
    sh_degree_interval: int = 1000
    random_background: bool = True

    # Densification (gsplat.strategy.DefaultStrategy), same defaults as nerfstudio's splatfacto.
    warmup_length: int = 500
    refine_every: int = 100
    reset_alpha_every: int = 30
    cull_alpha_thresh: float = 0.1
    cull_scale_thresh: float = 0.5
    cull_screen_size: float = 0.15
    densify_grad_thresh: float = 0.0008
    densify_size_thresh: float = 0.01
    split_screen_size: float = 0.05
    stop_split_at: int = 15_000
    stop_screen_size_at: int = 4_000

    model: GaussianModelConfig = field(default_factory=GaussianModelConfig)

    save_every: int = 5_000
    save_ply: bool = True
    log_every: int = 50
    device: str = "cuda"
    seed: int = 42


def _viewmat_from_camtoworld(camtoworld: torch.Tensor) -> torch.Tensor:
    """COLMAP poses are already in the OpenCV convention gsplat's rasterizer
    expects, so the world-to-camera matrix is a plain matrix inverse -- no
    axis-flip needed (contrast with nerfstudio, which stores OpenGL-convention
    poses and has to flip axes back before rasterizing)."""
    return torch.linalg.inv(camtoworld)


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = cfg.device

    scene = load_scene(cfg.data_dir, sparse_path=cfg.colmap_path, downscale_factor=cfg.downscale_factor)
    train_idx, eval_idx = train_eval_split(len(scene.image_names), cfg.eval_every)
    print(f"[data] {len(scene.image_names)} images ({len(train_idx)} train / {len(eval_idx)} eval), "
          f"{scene.points_xyz.shape[0]} SfM points, scene_scale={scene.scene_scale:.3f}")

    train_dataset = GaussianSplattingDataset(scene, train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=lambda x: x[0]
    )

    points_xyz = torch.from_numpy(scene.points_xyz).float().to(device)
    points_rgb = torch.from_numpy(scene.points_rgb).float().to(device) / 255.0
    params = init_gaussians(points_xyz, points_rgb, cfg.model, device=device)
    optimizers = build_optimizers(params, cfg.model, scene.scene_scale)
    means_scheduler = build_means_scheduler(optimizers, cfg.max_steps)

    strategy = DefaultStrategy(
        prune_opa=cfg.cull_alpha_thresh,
        grow_grad2d=cfg.densify_grad_thresh,
        grow_scale3d=cfg.densify_size_thresh,
        grow_scale2d=cfg.split_screen_size,
        prune_scale3d=cfg.cull_scale_thresh,
        prune_scale2d=cfg.cull_screen_size,
        refine_scale2d_stop_iter=cfg.stop_screen_size_at,
        refine_start_iter=cfg.warmup_length,
        refine_stop_iter=cfg.stop_split_at,
        reset_every=cfg.reset_alpha_every * cfg.refine_every,
        refine_every=cfg.refine_every,
        verbose=False,
    )
    strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale)

    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3).to(device)

    output_dir = Path(cfg.output_dir)
    ckpt_dir = output_dir / "checkpoints"

    def infinite_loader():
        while True:
            yield from train_loader

    data_iter = infinite_loader()
    pbar = tqdm(range(cfg.max_steps), desc="training")
    for step in pbar:
        batch = next(data_iter)
        camtoworld = batch["camtoworld"].to(device)[None]
        K = batch["K"].to(device)[None]
        width, height = batch["width"], batch["height"]
        gt_image = (batch["image"].to(device) / 255.0)[None]

        viewmat = _viewmat_from_camtoworld(camtoworld[0])[None]
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.model.sh_degree)

        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=get_scales(params),
            opacities=get_opacities(params),
            colors=get_colors(params),
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            packed=False,
            absgrad=strategy.absgrad,
        )

        strategy.step_pre_backward(params=params, optimizers=optimizers, state=strategy_state, step=step, info=info)

        if cfg.random_background:
            background = torch.rand(3, device=device)
        else:
            background = torch.zeros(3, device=device)
        pred_image = render[..., :3] + (1 - alpha) * background
        pred_image = torch.clamp(pred_image, 0.0, 1.0)

        l1 = F.l1_loss(pred_image, gt_image)
        ssim_loss = 1 - ssim_fn(pred_image.permute(0, 3, 1, 2), gt_image.permute(0, 3, 1, 2))
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * ssim_loss
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_scheduler.step()

        strategy.step_post_backward(params=params, optimizers=optimizers, state=strategy_state, step=step, info=info)

        if step % cfg.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", gaussians=params["means"].shape[0])

        if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.max_steps:
            ckpt_path = ckpt_dir / f"step-{step + 1:09d}.pt"
            save_checkpoint(
                ckpt_path,
                params,
                step + 1,
                extra={"sh_degree": cfg.model.sh_degree, "scene_scale": scene.scene_scale},
            )
            if cfg.save_ply:
                export_ply(output_dir / "point_cloud" / f"step-{step + 1:09d}.ply", params)
            tqdm.write(f"[checkpoint] step {step + 1}: {params['means'].shape[0]} gaussians -> {ckpt_path}")

    final_path = ckpt_dir / "final.pt"
    save_checkpoint(final_path, params, cfg.max_steps, extra={"sh_degree": cfg.model.sh_degree, "scene_scale": scene.scene_scale})
    return final_path
