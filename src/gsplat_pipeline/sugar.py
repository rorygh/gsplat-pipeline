"""Surface reconstruction from a trained Gaussian scene -- a compact,
Apache-licensed re-implementation of the ideas in SuGaR (Guedon & Lepetit,
CVPR 2024) built directly on `gsplat`, without vendoring the non-commercial
INRIA rasterizer (see docs/SURFACE_RECONSTRUCTION.md).

Two stages, both optional and both operating on an already-trained checkpoint
(so `train.py` is untouched):

1. **Surface alignment** (`sugar align`) -- a short refinement phase that adds
   SuGaR-style regularisers to the photometric loss:
     - *flatten*: drive each Gaussian's smallest scale toward zero (2D discs);
     - *opacify*: push opacities toward 1 (opaque shells, not fuzz);
     - *normal consistency*: align each Gaussian's thin axis with the surface
       normal implied by the rendered depth (the 2DGS/GOF trick, much cheaper
       than SuGaR's SDF sampling).
   Writes `<output>/surface.pt`.

2. **Mesh extraction** (`sugar mesh`) -- from the aligned (or raw) Gaussians:
     - *poisson*: sample points on the flat discs, normals = the thin axis
       oriented outward, screened Poisson (SuGaR's extraction step);
     - *tsdf*: render depth from every training view and TSDF-fuse (Open3D).
   Writes `<output>/mesh_<method>.ply` (+ `.obj`).

`sugar full` runs both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from pytorch_msssim import SSIM
from tqdm import tqdm

from .colmap.dataset import GaussianSplattingDataset, gaussians_in_masks, load_scene, train_eval_split
from .io import load_checkpoint, save_checkpoint
from .model import get_colors, get_opacities, get_scales


@dataclass
class SugarConfig:
    data_dir: Path
    checkpoint: Path
    output_dir: Path
    colmap_path: Optional[Path] = None
    downscale_factor: object = "auto"
    align: bool = True
    mask_dir: Optional[Path] = None

    # --- alignment refinement ---
    align_steps: int = 3000
    flatten_lambda: float = 0.1
    opacity_lambda: float = 0.02
    normal_lambda: float = 0.05
    ssim_lambda: float = 0.2

    # --- extraction ---
    method: Literal["poisson", "tsdf"] = "poisson"
    poisson_depth: int = 8
    disc_samples: int = 2          # extra points per Gaussian, on the disc plane
    opacity_keep: float = 0.35     # drop Gaussians below this opacity before sampling
    density_trim_quantile: float = 0.08
    tsdf_voxel: float = 0.01

    device: str = "cuda"
    seed: int = 42

    lrs: dict = field(default_factory=lambda: {
        "means": 1e-5, "scales": 5e-4, "quats": 1e-4, "opacities": 5e-3, "sh0": 1e-3, "shN": 5e-5,
    })


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def _thin_axis(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Unit vector along each Gaussian's smallest axis (its surface normal once
    the Gaussian is flat)."""
    R = _quat_to_rotmat(quats)
    idx = scales.argmin(dim=-1)
    return torch.gather(R, 2, idx[:, None, None].expand(-1, 3, 1)).squeeze(-1)


def _depth_to_normal(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Per-pixel surface normal in camera space from a depth map (H, W)."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                            torch.arange(w, device=depth.device, dtype=depth.dtype), indexing="ij")
    X = (xs - cx) / fx * depth
    Y = (ys - cy) / fy * depth
    P = torch.stack([X, Y, depth], dim=-1)
    dx = P[:, 2:, :] - P[:, :-2, :]
    dy = P[2:, :, :] - P[:-2, :, :]
    n = torch.cross(dx[1:-1, :, :], dy[:, 1:-1, :], dim=-1)
    n = F.normalize(n, dim=-1)
    return F.pad(n.permute(2, 0, 1), (1, 1, 1, 1)).permute(1, 2, 0)


# --------------------------------------------------------------------------
# stage 1: surface alignment
# --------------------------------------------------------------------------

def align_surface(cfg: SugarConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    scene = load_scene(cfg.data_dir, sparse_path=cfg.colmap_path, downscale_factor=cfg.downscale_factor,
                       align=cfg.align, mask_dir=cfg.mask_dir)
    masked = scene.mask_paths is not None
    train_idx, _ = train_eval_split(len(scene.image_names))
    ds = GaussianSplattingDataset(scene, train_idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=lambda x: x[0])

    params, meta = load_checkpoint(cfg.checkpoint, device=device)
    sh_degree = meta.get("sh_degree", 3)
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [params[k]], "lr": cfg.lrs[k]} for k in params])
    smax_init = get_scales(params).max(dim=-1).values.detach()
    scale_cap = float(torch.log(torch.tensor(0.25 * scene.scene_scale)))
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3).to(device)

    print(f"[sugar] aligning {params['means'].shape[0]} gaussians for {cfg.align_steps} steps")

    def batches():
        while True:
            yield from loader

    it = batches()
    pbar = tqdm(range(cfg.align_steps), desc="align")
    for _ in pbar:
        b = next(it)
        c2w = b["camtoworld"].to(device)
        K = b["K"].to(device)[None]
        w, h = int(b["width"]), int(b["height"])
        gt = (b["image"].to(device) / 255.0)[None]
        mask = b["mask"].to(device)[None] if masked else None
        viewmat = torch.linalg.inv(c2w)[None]

        scales = get_scales(params)
        opac = get_opacities(params)
        render, alpha, _ = rasterization(
            means=params["means"], quats=params["quats"], scales=scales, opacities=opac,
            colors=get_colors(params), viewmats=viewmat, Ks=K, width=w, height=h,
            sh_degree=sh_degree, render_mode="RGB+ED",
        )
        rgb, depth = render[..., :3], render[..., 3]
        bg = torch.rand(3, device=device)
        pred = torch.clamp(rgb + (1 - alpha) * bg, 0, 1)

        if masked:
            pred_c = pred * mask + bg * (1 - mask)
            gt_c = gt * mask + bg * (1 - mask)
        else:
            pred_c, gt_c = pred, gt
        l1 = F.l1_loss(pred_c, gt_c)
        ssim_l = 1 - ssim_fn(pred_c.permute(0, 3, 1, 2), gt_c.permute(0, 3, 1, 2))
        photo = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * ssim_l

        # SuGaR-style regularisers: drive the thinnest axis toward zero (flat
        # discs) without letting the other two run away.
        s_sorted = scales.sort(dim=-1).values
        smin, smax = s_sorted[:, 0], s_sorted[:, 2]
        flat = (smin / scene.scene_scale).mean() + 0.3 * F.relu(smax - smax_init).mean()
        opac_bin = torch.minimum(opac, 1 - opac).mean()

        normal_l = torch.zeros((), device=device)
        if cfg.normal_lambda > 0:
            gn = _thin_axis(params["quats"], scales.detach())  # (N,3) world
            gn_cam = gn @ viewmat[0, :3, :3].T
            nmap, _, _ = rasterization(
                means=params["means"], quats=params["quats"], scales=scales,
                opacities=opac, colors=gn_cam, viewmats=viewmat, Ks=K, width=w, height=h,
                sh_degree=None, render_mode="RGB",
            )
            nmap = F.normalize(nmap[0], dim=-1)
            dn = _depth_to_normal(depth[0].detach(), K[0])
            valid = (alpha[0, ..., 0] > 0.5)
            if mask is not None:
                valid = valid & (mask[0, ..., 0] > 0.5)
            if valid.any():
                agree = (nmap * dn).sum(-1).abs()[valid]
                normal_l = (1 - agree).mean()

        loss = photo + cfg.flatten_lambda * flat + cfg.opacity_lambda * opac_bin + cfg.normal_lambda * normal_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["scales"].clamp_(max=scale_cap)
        pbar.set_postfix(l1=f"{l1.item():.3f}", flat=f"{flat.item():.2f}", nrm=f"{float(normal_l):.2f}")

    if masked:
        keep = gaussians_in_masks(params["means"].detach().cpu().numpy(), scene, train_idx)
        keep_t = torch.from_numpy(keep).to(device)
        n0 = params["means"].shape[0]
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v[keep_t].detach()) for k, v in params.items()})
        print(f"[sugar] pruned to object mask: {n0} -> {params['means'].shape[0]} gaussians")

    out = Path(cfg.output_dir) / "surface.pt"
    save_checkpoint(out, params, meta.get("step", 0) + cfg.align_steps,
                    extra={"sh_degree": sh_degree, "scene_scale": scene.scene_scale,
                           "transform": scene.transform, "sugar_aligned": True})
    print(f"[sugar] wrote {out}")
    return out


# --------------------------------------------------------------------------
# stage 2: mesh extraction
# --------------------------------------------------------------------------

def _largest_component(mesh):
    import numpy as _np
    ti, nt, _ = mesh.cluster_connected_triangles()
    ti, nt = _np.asarray(ti), _np.asarray(nt)
    if len(nt):
        mesh.remove_triangles_by_mask(ti != int(nt.argmax()))
        mesh.remove_unreferenced_vertices()
    return mesh


@torch.no_grad()
def _poisson_mesh(cfg: SugarConfig, params, scene):
    import open3d as o3d

    means = params["means"].detach().cpu().numpy().astype(np.float64)
    quats = params["quats"].detach().cpu()
    scales = get_scales(params).detach().cpu()
    opac = get_opacities(params).detach().cpu().numpy()

    keep = opac > cfg.opacity_keep
    means, quats, scales = means[keep], quats[keep], scales[keep]
    R = _quat_to_rotmat(quats).numpy()
    order = np.argsort(scales.numpy(), axis=1)  # ascending; [:,0]=thin
    thin = np.take_along_axis(R, order[:, None, 0:1].repeat(3, 1), 2)[:, :, 0]
    ax1 = np.take_along_axis(R, order[:, None, 1:2].repeat(3, 1), 2)[:, :, 0]
    ax2 = np.take_along_axis(R, order[:, None, 2:3].repeat(3, 1), 2)[:, :, 0]
    s1 = np.take_along_axis(scales.numpy(), order[:, 1:2], 1)
    s2 = np.take_along_axis(scales.numpy(), order[:, 2:3], 1)

    pts = [means]
    nrm = [thin]
    rng = np.random.default_rng(0)
    for _ in range(cfg.disc_samples):
        u = rng.normal(size=(len(means), 1)) * s1
        v = rng.normal(size=(len(means), 1)) * s2
        pts.append(means + u * ax1 + v * ax2)
        nrm.append(thin)
    pts = np.concatenate(pts)
    nrm = np.concatenate(nrm)

    # orient normals outward from the object centroid
    centroid = means.mean(0)
    flip = np.sum(nrm * (pts - centroid), axis=1) < 0
    nrm[flip] *= -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=cfg.poisson_depth, linear_fit=True)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, cfg.density_trim_quantile))
    mesh = _largest_component(mesh)
    mesh.compute_vertex_normals()
    return mesh, len(pts)


@torch.no_grad()
def _tsdf_mesh(cfg: SugarConfig, params, scene):
    import open3d as o3d

    device = cfg.device
    sh_degree = 3
    train_idx, _ = train_eval_split(len(scene.image_names))
    ds = GaussianSplattingDataset(scene, train_idx)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.tsdf_voxel, sdf_trunc=cfg.tsdf_voxel * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for k in range(len(ds)):
        it = ds[k]
        c2w = it["camtoworld"].to(device)
        K = it["K"].to(device)[None]
        w, h = int(it["width"]), int(it["height"])
        out, alpha, _ = rasterization(
            means=params["means"], quats=params["quats"], scales=get_scales(params),
            opacities=get_opacities(params), colors=get_colors(params),
            viewmats=torch.linalg.inv(c2w)[None], Ks=K, width=w, height=h,
            sh_degree=sh_degree, render_mode="RGB+ED")
        rgb = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth = out[0, ..., 3].cpu().numpy().astype(np.float32)
        depth[alpha[0, ..., 0].cpu().numpy() < 0.5] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0, depth_trunc=float(np.percentile(depth[depth > 0], 99) * 1.5) if (depth > 0).any() else 10.0,
            convert_rgb_to_intensity=False)
        Kn = K[0].cpu().numpy()
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2])
        vol.integrate(rgbd, intr, np.linalg.inv(c2w.cpu().numpy()))
    mesh = _largest_component(vol.extract_triangle_mesh())
    mesh.compute_vertex_normals()
    return mesh, len(ds)


def extract_mesh(cfg: SugarConfig, checkpoint: Optional[Path] = None) -> Path:
    import time

    import open3d as o3d

    device = cfg.device
    scene = load_scene(cfg.data_dir, sparse_path=cfg.colmap_path, downscale_factor=cfg.downscale_factor,
                       align=cfg.align, mask_dir=cfg.mask_dir)
    params, _ = load_checkpoint(checkpoint or cfg.checkpoint, device=device)

    t0 = time.time()
    if cfg.method == "poisson":
        mesh, n = _poisson_mesh(cfg, params, scene)
        detail = f"{n} sample points"
    else:
        mesh, n = _tsdf_mesh(cfg, params, scene)
        detail = f"{n} views fused"

    out = Path(cfg.output_dir) / f"mesh_{cfg.method}.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh)
    o3d.io.write_triangle_mesh(str(out.with_suffix(".obj")), mesh)
    print(f"[sugar] {cfg.method}: {detail} -> {len(mesh.vertices)} verts / {len(mesh.triangles)} tris, "
          f"watertight={mesh.is_watertight()}, {time.time() - t0:.1f}s -> {out}")
    return out


def run_sugar(cfg: SugarConfig, do_align: bool = True, do_mesh: bool = True) -> None:
    ckpt = cfg.checkpoint
    if do_align:
        ckpt = align_surface(cfg)
    if do_mesh:
        extract_mesh(cfg, checkpoint=ckpt)
