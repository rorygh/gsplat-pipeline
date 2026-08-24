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
) -> None:
    """Run COLMAP structure-from-motion on a folder of images."""
    from .colmap.runner import run_sfm

    model_dir = run_sfm(image_dir, output_dir, matching_method=matching_method, camera_model=camera_model)
    print(f"[sfm] sparse model written to {model_dir}")


@app.command
def train(
    data_dir: Path,
    output_dir: Path,
    colmap_path: Optional[Path] = None,
    max_steps: int = 30_000,
    device: str = "cuda",
) -> None:
    """Train a Gaussian Splatting scene from a COLMAP reconstruction."""
    from .train import TrainConfig
    from .train import train as run_train

    cfg = TrainConfig(data_dir=data_dir, output_dir=output_dir, colmap_path=colmap_path, max_steps=max_steps, device=device)
    ckpt = run_train(cfg)
    print(f"[train] final checkpoint: {ckpt}")


@app.command(name="eval")
def eval_cmd(
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    colmap_path: Optional[Path] = None,
    device: str = "cuda",
) -> None:
    """Evaluate a trained checkpoint on its held-out views (PSNR/SSIM + renders)."""
    from .eval import EvalConfig, evaluate

    cfg = EvalConfig(data_dir=data_dir, checkpoint=checkpoint, output_dir=output_dir, colmap_path=colmap_path, device=device)
    evaluate(cfg)


@app.command
def view(checkpoint: Path, port: int = 7007, device: str = "cuda") -> None:
    """Launch the interactive viewer for a trained checkpoint."""
    from .viewer import ViewerConfig, run_viewer

    run_viewer(ViewerConfig(checkpoint=checkpoint, port=port, device=device))


def main() -> None:
    app.cli()


if __name__ == "__main__":
    main()
