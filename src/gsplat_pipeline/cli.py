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
) -> None:
    """Run COLMAP structure-from-motion on a folder of images.

    use_gpu: set False for a CPU-only COLMAP build (e.g. conda-forge's
    `cpu_*` build on a headless pod), otherwise feature extraction/matching
    abort with no CUDA/GL context.
    """
    from .colmap.runner import run_sfm

    model_dir = run_sfm(
        image_dir, output_dir, matching_method=matching_method, camera_model=camera_model, use_gpu=use_gpu
    )
    print(f"[sfm] sparse model written to {model_dir}")


@app.command
def train(
    data_dir: Path,
    output_dir: Path,
    colmap_path: Optional[Path] = None,
    downscale_factor: Optional[int] = None,
    max_steps: int = 30_000,
    device: str = "cuda",
) -> None:
    """Train a Gaussian Splatting scene from a COLMAP reconstruction.

    downscale_factor: an explicit power of 2, or unset to auto-pick (keeps the
    long edge under 1600px, using a pre-existing images_{factor}/ folder if
    present -- matches nerfstudio's ColmapDataParser default behavior).
    """
    from .train import TrainConfig
    from .train import train as run_train

    cfg = TrainConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        colmap_path=colmap_path,
        downscale_factor=downscale_factor if downscale_factor is not None else "auto",
        max_steps=max_steps,
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
    device: str = "cuda",
) -> None:
    """Evaluate a trained checkpoint on its held-out views (PSNR/SSIM + renders)."""
    from .eval import EvalConfig, evaluate

    cfg = EvalConfig(
        data_dir=data_dir,
        checkpoint=checkpoint,
        output_dir=output_dir,
        colmap_path=colmap_path,
        downscale_factor=downscale_factor if downscale_factor is not None else "auto",
        device=device,
    )
    evaluate(cfg)


@app.command
def mask(
    image_dir: Path,
    output_dir: Path,
    model: str = "birefnet-general",
    composite: Optional[str] = "white",
    alpha_matting: bool = False,
    feather: int = 2,
) -> None:
    """Background removal (v0): write per-image foreground masks for an
    object-centric capture, using a salient-object segmentation model (rembg).

    Appearance-based and per-frame -- good when one object dominates each
    frame. See docs/BACKGROUND_REMOVAL_PLAN.md for the moving-object case.
    """
    from .masking import MaskConfig, run_masking

    run_masking(MaskConfig(
        image_dir=image_dir, output_dir=output_dir, model=model,
        composite=composite, alpha_matting=alpha_matting, feather=feather,
    ))


@app.command(name="colmap-report")
def colmap_report(
    data_dir: Path,
    output_dir: Path,
    colmap_path: Optional[Path] = None,
    image_dir: Optional[Path] = None,
) -> None:
    """Sanity-check a COLMAP sparse model: camera-path plots, per-image
    connectivity, and reprojection-error distribution (PNGs + JSON).

    data_dir: a COLMAP reconstruction dir (auto-detects sparse/0, colmap/sparse/0);
    or pass --colmap-path to point straight at the model folder.
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

    write_report(sparse_path, output_dir, image_dir=image_dir)


@app.command
def view(checkpoint: Path, port: int = 7007, device: str = "cuda") -> None:
    """Launch the interactive viewer for a trained checkpoint."""
    from .viewer import ViewerConfig, run_viewer

    run_viewer(ViewerConfig(checkpoint=checkpoint, port=port, device=device))


def main() -> None:
    app.cli()


if __name__ == "__main__":
    main()
