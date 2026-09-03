"""Background removal (v0): per-image foreground masks for object-centric
captures.

This is the appearance-based path -- a salient-object / dichotomous
segmentation model (via `rembg`) picks the dominant central object out of
each frame independently. It has no notion of motion or multi-view
consistency; it just asks "what's the subject of this photo". That's enough
when the object dominates the frame (a walkaround of one car, a turntable
capture) and is the right first thing to try.

For the harder case -- object moving, background static -- see
docs/BACKGROUND_REMOVAL_PLAN.md; that needs motion cues this module doesn't
use.

Masks are written as 8-bit PNGs (255 = keep, 0 = drop) named to match the
source images, so a downstream consumer can `mask/<name>.png` alongside
`images/<name>`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class MaskConfig:
    image_dir: Path
    output_dir: Path
    model: str = "birefnet-general"
    """rembg session name. birefnet-general = high-res dichotomous segmentation
    (good default); isnet-general-use = lighter/faster; u2net = original."""
    composite: Optional[str] = "white"
    """also write images_masked/ with the background flattened to this colour
    ("white", "black", or None to skip). Matches the 3DGS convention of
    training against a white background outside the mask."""
    alpha_matting: bool = False
    """rembg alpha matting -- cleaner hair/foliage edges, ~3x slower."""
    feather: int = 2
    """gaussian-blur the mask edge by this many px (0 = hard edge)."""
    min_area_frac: float = 0.01
    """warn if the kept region is smaller than this fraction of the frame
    (usually means the model latched onto the wrong thing)."""


def _load_session(model: str):
    try:
        from rembg import new_session
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "background removal needs `rembg` -- `pip install gsplat-pipeline[masks]`"
        ) from e
    return new_session(model)


def run_masking(cfg: MaskConfig) -> dict:
    import cv2
    from rembg import remove

    session = _load_session(cfg.model)

    mask_dir = Path(cfg.output_dir) / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = None
    if cfg.composite:
        comp_dir = Path(cfg.output_dir) / "images_masked"
        comp_dir.mkdir(parents=True, exist_ok=True)
    fill = 255 if cfg.composite == "white" else 0

    images = sorted(p for p in Path(cfg.image_dir).iterdir() if p.suffix in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"no images in {cfg.image_dir}")

    areas, suspicious = [], []
    for path in images:
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        cutout = remove(
            rgb, session=session, alpha_matting=cfg.alpha_matting,
            only_mask=True, post_process_mask=True,
        )
        mask = np.asarray(cutout)
        if mask.ndim == 3:
            mask = mask[..., -1]

        if cfg.feather > 0:
            k = cfg.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        area_frac = float((mask > 127).mean())
        areas.append(area_frac)
        if area_frac < cfg.min_area_frac:
            suspicious.append(path.name)

        cv2.imwrite(str(mask_dir / f"{path.stem}.png"), mask)

        if comp_dir is not None:
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            out = (bgr.astype(np.float32) * alpha + fill * (1.0 - alpha)).astype(np.uint8)
            cv2.imwrite(str(comp_dir / path.name), out)

    areas = np.array(areas)
    result = {
        "num_images": len(images),
        "model": cfg.model,
        "mean_foreground_frac": float(areas.mean()),
        "min_foreground_frac": float(areas.min()),
        "suspicious_images": suspicious,
        "mask_dir": str(mask_dir),
    }
    print(f"[mask] {len(images)} images -> {mask_dir} "
          f"(foreground {areas.mean():.1%} mean, {areas.min():.1%} min)")
    if suspicious:
        print(f"[mask] {len(suspicious)} image(s) with a tiny foreground -- check these: "
              f"{', '.join(suspicious[:8])}{' ...' if len(suspicious) > 8 else ''}")
    if comp_dir is not None:
        print(f"[mask] composited images -> {comp_dir}")
    return result
