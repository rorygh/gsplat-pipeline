"""Run COLMAP's SfM pipeline on a folder of images, producing a sparse model.

A thin subprocess wrapper around the `colmap` CLI: feature extraction ->
matching -> incremental mapping. No video/frame-extraction, mask handling, or
vocab-tree matching support -- if you need those, use nerfstudio's
`ns-process-data` instead. This covers the common case: a folder of photos
with real overlap in, a `sparse/0/{cameras,images,points3D}.bin` model out.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _use_gpu_flag(binary: str) -> bool:
    """COLMAP renamed --SiftExtraction.use_gpu -> --FeatureExtraction.use_gpu (and the
    matching equivalent) at some point after 4.0. Probe the installed binary's own
    subcommand help rather than hardcoding a version cutoff -- `colmap -h` alone only
    lists subcommand names, not their flags, so this has to ask `feature_extractor -h`
    specifically. COLMAP prints help to stderr, not stdout."""
    help_text = subprocess.run([binary, "feature_extractor", "-h"], capture_output=True, text=True).stderr
    return "FeatureExtraction.use_gpu" in help_text


def run_sfm(
    image_dir: Path,
    output_dir: Path,
    matching_method: Literal["exhaustive", "sequential"] = "exhaustive",
    use_gpu: bool = True,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
) -> Path:
    """Run COLMAP feature extraction, matching, and mapping.

    Returns the path to the resulting sparse model (`<output_dir>/sparse/0`).
    """
    if shutil.which("colmap") is None:
        raise RuntimeError("`colmap` not found on PATH. Install it (see README) before running SfM.")

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    new_style = _use_gpu_flag("colmap")
    extract_gpu_flag = f"--{'FeatureExtraction' if new_style else 'SiftExtraction'}.use_gpu"
    match_gpu_flag = f"--{'FeatureMatching' if new_style else 'SiftMatching'}.use_gpu"

    _run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            extract_gpu_flag,
            "1" if use_gpu else "0",
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.single_camera",
            "1" if single_camera else "0",
        ]
    )

    matcher_cmd = "exhaustive_matcher" if matching_method == "exhaustive" else "sequential_matcher"
    _run(
        [
            "colmap",
            matcher_cmd,
            "--database_path",
            str(database_path),
            match_gpu_flag,
            "1" if use_gpu else "0",
        ]
    )

    _run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
        ]
    )

    model_dir = sparse_dir / "0"
    if not (model_dir / "cameras.bin").exists():
        raise RuntimeError(
            f"COLMAP mapper did not produce a model at {model_dir}. "
            "Check that the input images have enough overlap to register."
        )
    return model_dir
