"""Checkpoint save/load and .ply export for a trained Gaussian scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def save_checkpoint(path: Path, params: torch.nn.ParameterDict, step: int, extra: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "params": {k: v.detach().cpu() for k, v in params.items()},
        **(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cuda") -> tuple[torch.nn.ParameterDict, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in payload["params"].items()})
    meta = {k: v for k, v in payload.items() if k != "params"}
    return params, meta


def export_ply(path: Path, params: torch.nn.ParameterDict) -> None:
    """Write a standard 3DGS-viewer-compatible .ply (the same field layout used by
    the original INRIA 3DGS codebase and nerfstudio's splatfacto exporter), so the
    trained scene can be opened in any off-the-shelf Gaussian Splat viewer."""
    means = params["means"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()
    opacities = params["opacities"].detach().cpu().numpy()  # stored pre-sigmoid, matches convention below
    sh0 = params["sh0"].detach().cpu().numpy().reshape(means.shape[0], -1)
    shN = params["shN"].detach().cpu().numpy().reshape(means.shape[0], -1)

    n = means.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    dtype += [(f"f_dc_{i}", "f4") for i in range(sh0.shape[1])]
    dtype += [(f"f_rest_{i}", "f4") for i in range(shN.shape[1])]
    dtype += [("opacity", "f4")]
    dtype += [(f"scale_{i}", "f4") for i in range(3)]
    dtype += [(f"rot_{i}", "f4") for i in range(4)]

    elements = np.empty(n, dtype=dtype)
    elements["x"], elements["y"], elements["z"] = means[:, 0], means[:, 1], means[:, 2]
    elements["nx"], elements["ny"], elements["nz"] = 0.0, 0.0, 0.0
    for i in range(sh0.shape[1]):
        elements[f"f_dc_{i}"] = sh0[:, i]
    for i in range(shN.shape[1]):
        elements[f"f_rest_{i}"] = shN[:, i]
    elements["opacity"] = opacities
    for i in range(3):
        elements[f"scale_{i}"] = params["scales"].detach().cpu().numpy()[:, i]
    for i in range(4):
        elements[f"rot_{i}"] = quats[:, i]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            + "".join(f"property float {name}\n" for name, _ in dtype)
            + "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(elements.tobytes())
