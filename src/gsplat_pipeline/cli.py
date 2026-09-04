"""Command-line entry point: `gsplat-pipeline sfm|train|eval|view`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from tyro.extras import SubcommandApp

app = SubcommandApp()


@app.command
def sfm(
    image_dir: Path,
    output_dir: Path,
    matching_method: Literal["exhaustive", "sequential"] = "exhaustive",
    camera_model: str = "OPENCV",
    use_gpu: bool = True,
    mask_path: Optional[Path] = None,
) -> None:
    """Run COLMAP structure-from-motion on a folder of images.

    use_gpu: set False for a CPU-only COLMAP build (e.g. conda-forge's
    `cpu_*` build on a headless pod), otherwise feature extraction/matching
    abort with no CUDA/GL context.

    mask_path: directory of COLMAP-format masks (`<image_name>.png`, black =
    ignore) to restrict keypoint detection. Produce it with
    `gsplat-pipeline mask --colmap-naming` (add `--invert` for a moving object).
    """
    from .colmap.runner import run_sfm

    model_dir = run_sfm(
        image_dir, output_dir, matching_method=matching_method, camera_model=camera_model,
        use_gpu=use_gpu, mask_path=mask_path,
    )
    print(f"[sfm] sparse model written to {model_dir}")


@app.command
def train(
    data_dir: Path,
    output_dir: Path,
    colmap_path: Optional[Path] = None,
    downscale_factor: Optional[int] = None,
    max_steps: int = 30_000,
    align: bool = True,
    mask_dir: Optional[Path] = None,
    device: str = "cuda",
) -> None:
    """Train a Gaussian Splatting scene from a COLMAP reconstruction.

    mask_dir: directory of per-image object masks (`<stem>.png`, 255 = object,
    e.g. from `gsplat-pipeline mask`). Drops off-object SfM points, restricts the
    loss to the object, supervises alpha, and prunes the final model to the
    object -- so the result is the segmented object only.

    downscale_factor: an explicit power of 2, or unset to auto-pick (keeps the
    long edge under 1600px, using a pre-existing images_{factor}/ folder if
    present -- matches nerfstudio's ColmapDataParser default behavior).

    align: bake a +Z-up orbit frame (recovered from the camera trajectory) into
    the checkpoint and PLY. Assumes a roughly planar orbit; --no-align keeps
    raw COLMAP world axes.
    """
    from .train import TrainConfig
    from .train import train as run_train

    cfg = TrainConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        colmap_path=colmap_path,
        downscale_factor=downscale_factor if downscale_factor is not None else "auto",
        max_steps=max_steps,
        align=align,
        mask_dir=mask_dir,
        device=device,
    )
    ckpt = run_train(cfg)
    print(f"[train] final checkpoint: {ckpt}")


@app.command(name="eval")
def eval_cmd(
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    colmap_path: Optional[Path] = None,
    downscale_factor: Optional[int] = None,
    align: bool = True,
    mask_dir: Optional[Path] = None,
    device: str = "cuda",
) -> None:
    """Evaluate a trained checkpoint on its held-out views (PSNR/SSIM + renders).

    align: must match what the checkpoint was trained with (default on).
    mask_dir: if the checkpoint was mask-trained, pass the same masks so metrics
    are computed on the object region only.
    """
    from .eval import EvalConfig, evaluate

    cfg = EvalConfig(
        data_dir=data_dir,
        checkpoint=checkpoint,
        output_dir=output_dir,
        colmap_path=colmap_path,
        downscale_factor=downscale_factor if downscale_factor is not None else "auto",
        align=align,
        mask_dir=mask_dir,
        device=device,
    )
    evaluate(cfg)


@app.command
def mask(
    image_dir: Path,
    output_dir: Path = Path("masks"),
    prompt: Optional[str] = None,
    model: str = "birefnet-general",
    threshold: float = 0.5,
    temporal: bool = True,
    temporal_window: int = 2,
    alpha_matting: bool = False,
    composite: Optional[str] = "white",
    feather: int = 2,
    invert: bool = False,
    colmap_naming: bool = False,
    contact_sheet: bool = True,
) -> None:
    """Background removal (v0.2): write per-image foreground masks for an
    object-centric capture.

    Raw masks come from a salient-object model (rembg/BiRefNet) or, with
    --prompt "a car", from CLIPSeg (segment what you name). --temporal (on by
    default) then runs an optical-flow pass over the ordered sequence to
    remove per-frame flicker. Outputs default to ./masks/ (gitignored).

    See docs/BACKGROUND_REMOVAL_PLAN.md for the moving-object case.
    """
    from .masking import MaskConfig, run_masking

    run_masking(MaskConfig(
        image_dir=image_dir, output_dir=output_dir, prompt=prompt, model=model,
        threshold=threshold, temporal=temporal, temporal_window=temporal_window,
        alpha_matting=alpha_matting, composite=composite, feather=feather,
        invert=invert, colmap_naming=colmap_naming,
        contact_sheet=contact_sheet,
    ))


@app.command(name="colmap-report")
def colmap_report(
    data_dir: Path,
    output_dir: Path,
    colmap_path: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    align: bool = True,
) -> None:
    """Sanity-check a COLMAP sparse model: camera-path plots, per-image
    connectivity, and reprojection-error distribution (PNGs + JSON).

    data_dir: a COLMAP reconstruction dir (auto-detects sparse/0, colmap/sparse/0);
    or pass --colmap-path to point straight at the model folder.

    align: rotate the plots into the +Z-up orbit frame (assumes a roughly
    planar orbit), matching what `train`/`view` bake into the scene. --no-align
    keeps raw COLMAP world axes.
    """
    from .colmap.report import write_report

    sparse_path = colmap_path
    if sparse_path is None:
        for candidate in ("sparse/0", "colmap/sparse/0", "sparse"):
            if (data_dir / candidate / "cameras.bin").exists():
                sparse_path = data_dir / candidate
                break
        else:
            raise FileNotFoundError(f"No COLMAP sparse model under {data_dir}")
    if image_dir is None and (data_dir / "images").is_dir():
        image_dir = data_dir / "images"

    write_report(sparse_path, output_dir, image_dir=image_dir, align=align)


@app.command
def view(checkpoint: Path, port: int = 7007, device: str = "cuda") -> None:
    """Launch the interactive viewer for a trained checkpoint."""
    from .viewer import ViewerConfig, run_viewer

    run_viewer(ViewerConfig(checkpoint=checkpoint, port=port, device=device))


@app.command
def sugar(
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    stage: Literal["full", "align", "mesh"] = "full",
    method: Literal["poisson", "tsdf"] = "poisson",
    colmap_path: Optional[Path] = None,
    align: bool = True,
    mask_dir: Optional[Path] = None,
    align_steps: int = 3000,
    poisson_depth: int = 9,
    tsdf_voxel: float = 0.01,
    device: str = "cuda",
) -> None:
    """Extract a surface mesh from a trained checkpoint -- a compact,
    Apache-licensed take on SuGaR (docs/SURFACE_RECONSTRUCTION.md). Does not
    touch `train`.

    stage=align: short refinement that flattens Gaussians onto the surface
    (SuGaR regularisers) -> <output>/surface.pt.
    stage=mesh:  extract from the checkpoint as-is (poisson = sample the discs +
    screened Poisson; tsdf = fuse rendered depth).
    stage=full:  align, then mesh the aligned model.

    Pair with --mask-dir (same masks as training) for a clean object mesh.
    """
    from .sugar import SugarConfig, run_sugar

    cfg = SugarConfig(
        data_dir=data_dir, checkpoint=checkpoint, output_dir=output_dir, method=method,
        colmap_path=colmap_path, align=align, mask_dir=mask_dir, align_steps=align_steps,
        poisson_depth=poisson_depth, tsdf_voxel=tsdf_voxel, device=device,
    )
    run_sugar(cfg, do_align=stage in ("full", "align"), do_mesh=stage in ("full", "mesh"))


def main() -> None:
    app.cli()


if __name__ == "__main__":
    main()
