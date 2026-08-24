"""Interactive Gaussian Splat viewer, built directly on `viser`.

Viser's client camera (`wxyz`/`position`) already follows OpenCV
conventions -- the same convention COLMAP and this pipeline's `camtoworlds`
use throughout -- so no axis-flipping is needed here (unlike nerfstudio,
whose internal camera convention is OpenGL-based and has to flip axes back
before rasterizing).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import viser
from gsplat.rendering import rasterization

from .io import load_checkpoint
from .model import get_colors, get_opacities, get_scales

MAX_RENDER_WIDTH = 1600
VISER_DEFAULT_BACKGROUND = (0.1490, 0.1647, 0.2157)


@dataclass
class ViewerConfig:
    checkpoint: Path
    device: str = "cuda"
    port: int = 7007


@torch.no_grad()
def render_from_camera(
    params: torch.nn.ParameterDict,
    sh_degree: int,
    device: str,
    wxyz: np.ndarray,
    position: np.ndarray,
    fov: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Render the scene from a viser camera pose. Pure function (no viser
    server dependency) so it can be exercised directly in tests."""
    fy = height / 2.0 / np.tan(fov / 2.0)
    fx = fy  # square pixels
    K = torch.tensor([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]], dtype=torch.float32, device=device)[
        None
    ]

    R = viser.transforms.SO3(wxyz).as_matrix()
    camtoworld = np.eye(4)
    camtoworld[:3, :3] = R
    camtoworld[:3, 3] = position
    camtoworld = torch.from_numpy(camtoworld).float().to(device)
    viewmat = torch.linalg.inv(camtoworld)[None]

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
    background = torch.tensor(VISER_DEFAULT_BACKGROUND, device=device)
    image = torch.clamp(render[0, ..., :3] + (1 - alpha[0]) * background, 0.0, 1.0)
    return image.cpu().numpy()


class _ClientRenderLoop:
    """One background thread per connected client: renders whenever that
    client's camera has moved since the last frame."""

    def __init__(self, client: viser.ClientHandle, params: torch.nn.ParameterDict, sh_degree: int, device: str):
        self.client = client
        self.params = params
        self.sh_degree = sh_degree
        self.device = device
        self._dirty = threading.Event()
        self._dirty.set()
        self._stop = threading.Event()
        client.camera.on_update(lambda _: self._dirty.set())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._dirty.wait(timeout=0.2):
                continue
            self._dirty.clear()
            camera = self.client.camera

            aspect = camera.aspect
            width = min(MAX_RENDER_WIDTH, camera.image_width if camera.image_width else 1280)
            height = int(width / aspect)

            image = render_from_camera(
                self.params, self.sh_degree, self.device, camera.wxyz, camera.position, camera.fov, width, height
            )
            self.client.scene.set_background_image(image, format="jpeg")


def run_viewer(cfg: ViewerConfig) -> None:
    params, meta = load_checkpoint(cfg.checkpoint, device=cfg.device)
    sh_degree = meta.get("sh_degree", 3)
    step = meta.get("step", "?")
    print(f"[viewer] loaded checkpoint at step {step}, {params['means'].shape[0]} gaussians")

    server = viser.ViserServer(port=cfg.port)
    loops: list[_ClientRenderLoop] = []

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        loops.append(_ClientRenderLoop(client, params, sh_degree, cfg.device))

    print(f"[viewer] serving at http://localhost:{cfg.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        for loop in loops:
            loop.stop()
