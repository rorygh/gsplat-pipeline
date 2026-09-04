"""Background removal (v0.2): per-image foreground masks for object-centric
captures, with temporal stabilization and an optional text prompt.

Two ways to get the raw per-frame mask:

1. **Saliency** (default) -- rembg / BiRefNet picks the dominant central
   object in each frame independently ("what is the subject of this photo").
2. **Text prompt** (``--prompt "a car"``) -- CLIPSeg segments whatever the
   phrase names, so an off-centre or non-dominant object still works and an
   ambiguous scene (two cars, "the *red* one") can be steered by wording.

On top of either, ``--temporal`` (on by default) runs an optical-flow pass
over the ordered frame sequence: each mask is re-estimated as a vote between
itself and its flow-warped neighbours, and any frame whose area jumps far
from the local median is rebuilt from neighbours alone. This removes the
per-frame flicker / latch-on that both single-frame models produce when the
object gets small, blurred, or partly occluded -- the failure visible in
``colmap-report`` as wildly varying observation counts.

For the harder case -- object moving, background static -- see
docs/BACKGROUND_REMOVAL_PLAN.md; that needs motion cues this module doesn't
use.

Masks are written as 8-bit PNGs (255 = keep, 0 = drop) named to match the
source images, so a downstream consumer can pair ``mask/<stem>.png`` with
``images/<name>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"


@dataclass
class MaskConfig:
    image_dir: Path
    output_dir: Path

    # --- raw per-frame segmentation ---
    prompt: Optional[str] = None
    """text prompt for semantic segmentation via CLIPSeg (e.g. "a car",
    "the silver sedan"). When set, the raw mask comes from CLIPSeg instead of
    the rembg saliency model, and `model` is ignored."""
    model: str = "birefnet-general"
    """rembg session name for the saliency path (used when `prompt` is unset).
    birefnet-general = high-res dichotomous segmentation (good default);
    isnet-general-use = lighter/faster; u2net = original."""
    threshold: float = 0.5
    """probability threshold turning the soft mask into a binary one. Lower it
    (~0.3) for a text prompt if the object is under-segmented."""
    alpha_matting: bool = False
    """rembg alpha matting -- cleaner hair/foliage edges, ~3x slower (saliency
    path only)."""

    # --- temporal stabilization ---
    temporal: bool = True
    """smooth the mask sequence with optical-flow propagation + voting.
    Assumes images sorted by filename are an ordered capture."""
    temporal_window: int = 2
    """half-width (frames each side) of the temporal voting window."""
    temporal_area_gate: float = 0.5
    """if a frame's mask area deviates from its local median by more than this
    fraction, distrust that frame and rebuild it from warped neighbours."""
    flow_max_width: int = 1024
    """cap the width at which optical flow is computed (speed / robustness);
    the stabilized mask is upsampled back to full resolution afterwards."""

    # --- output ---
    composite: Optional[str] = "white"
    """also write images_masked/ with the background flattened to this colour
    ("white", "black", or None to skip). Matches the 3DGS convention of
    training against a constant background outside the mask."""
    feather: int = 2
    """gaussian-blur the mask edge by this many px (0 = hard edge)."""
    invert: bool = False
    """write the complement (object black, background white). Use for the SfM
    pass on a *moving* object: pose estimation should ignore the object."""
    colmap_naming: bool = False
    """name outputs `<image_name>.png` (e.g. frame_0001.jpg.png) instead of
    `<stem>.png`, so the folder can be passed straight to `sfm --mask-path`."""
    contact_sheet: bool = True
    """write contact_sheet.jpg -- every frame as a thumbnail with the mask
    outlined and the background dimmed, for eyeballing temporal stability."""
    min_area_frac: float = 0.01
    """warn if the kept region is smaller than this fraction of the frame
    (usually means the model latched onto the wrong thing)."""


# --------------------------------------------------------------------------
# raw per-frame maskers
# --------------------------------------------------------------------------

def _preload_cuda_libs() -> None:
    """onnxruntime-gpu's CUDA provider dlopens cuDNN/cuBLAS by soname; make the
    copies shipped in the `nvidia-*-cu12` wheels (pulled in by the CUDA torch
    build) resolvable by loading them into the global symbol namespace first.
    Best-effort -- if anything is missing the session just falls back to CPU."""
    import ctypes
    import glob
    import importlib.util

    groups = (
        ("nvidia.cublas", ("libcublasLt.so.*", "libcublas.so.*")),
        ("nvidia.cudnn", ("libcudnn_graph.so.*", "libcudnn_engines_precompiled.so.*",
                          "libcudnn_engines_runtime_compiled.so.*", "libcudnn_heuristic.so.*",
                          "libcudnn_ops.so.*", "libcudnn_adv.so.*", "libcudnn_cnn.so.*",
                          "libcudnn.so.*")),
    )
    for pkg, patterns in groups:
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        libdir = Path(spec.submodule_search_locations[0]) / "lib"
        for pat in patterns:
            for so in sorted(glob.glob(str(libdir / pat))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _load_session(model: str):
    try:
        import onnxruntime as ort
        from rembg import new_session
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "background removal needs `rembg` -- `pip install gsplat-pipeline[masks]`"
        ) from e

    available = ort.get_available_providers()
    # Prefer CUDA when an onnxruntime-gpu build is installed; skip TensorRT
    # (long per-shape build, and rembg feeds it varying input sizes).
    if "CUDAExecutionProvider" in available:
        _preload_cuda_libs()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = new_session(model, providers=providers)
    active = session.inner_session.get_providers()
    if "CUDAExecutionProvider" in active:
        print("[mask] rembg on GPU (CUDAExecutionProvider)")
    else:
        print("[mask] rembg on CPU -- `pip install onnxruntime-gpu` (matching CUDA) for a large speedup")
    return session


def _saliency_masker(model: str, alpha_matting: bool) -> Callable[[np.ndarray], np.ndarray]:
    from rembg import remove

    session = _load_session(model)

    def infer(rgb: np.ndarray) -> np.ndarray:
        cutout = remove(
            rgb, session=session, alpha_matting=alpha_matting,
            only_mask=True, post_process_mask=True,
        )
        mask = np.asarray(cutout)
        if mask.ndim == 3:
            mask = mask[..., -1]
        return mask.astype(np.float32) / 255.0

    return infer


def _clipseg_masker(prompt: str) -> Callable[[np.ndarray], np.ndarray]:
    try:
        import torch
        from transformers import AutoProcessor, CLIPSegForImageSegmentation
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "text-prompt masking needs `transformers` -- "
            "`pip install gsplat-pipeline[masks-text]`"
        ) from e

    processor = AutoProcessor.from_pretrained(CLIPSEG_MODEL)
    model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    @torch.no_grad()
    def infer(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        inputs = processor(text=[prompt], images=[rgb], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits.float()  # (H', W') or (1, H', W')
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_CUBIC)

    return infer


# --------------------------------------------------------------------------
# binary-mask cleanup
# --------------------------------------------------------------------------

def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected foreground blob."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if num <= 2:  # background + at most one component
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill background pockets fully enclosed by the foreground. Pad a 1px
    background border so the outside is one connected region regardless of the
    object touching an edge, flood it away, and keep what's left."""
    h, w = binary.shape
    flood = np.pad((~binary).astype(np.uint8), 1, constant_values=1)  # 1 = background
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)
    return binary | flood[1:-1, 1:-1].astype(bool)


def _clean_binary(prob: np.ndarray, threshold: float) -> np.ndarray:
    binary = prob >= threshold
    if not binary.any():
        return binary
    binary = _largest_component(binary)
    return _fill_holes(binary)


# --------------------------------------------------------------------------
# optical-flow temporal stabilization
# --------------------------------------------------------------------------

def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample `img` at each pixel displaced by `flow` (bilinear, edge-clamped)."""
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx = xs + flow[..., 0]
    mapy = ys + flow[..., 1]
    return cv2.remap(
        img.astype(np.float32), mapx, mapy,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def _compose(flow_ab: np.ndarray, flow_bc: np.ndarray) -> np.ndarray:
    """Chain two flow fields: result maps frame a's pixels to frame c."""
    return flow_ab + _warp(flow_bc, flow_ab)


def _stabilize(
    probs: list[np.ndarray],
    grays: list[np.ndarray],
    window: int,
    area_gate: float,
    decay: float = 0.6,
) -> list[np.ndarray]:
    """Re-estimate each soft mask as a flow-weighted vote over a temporal
    window. Frames whose area deviates far from the local median are dropped
    from their own vote and rebuilt from neighbours."""
    n = len(probs)
    if n < 3 or window < 1:
        return probs

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    step_fwd = [dis.calc(grays[i], grays[i + 1], None) for i in range(n - 1)]  # i -> i+1
    step_bwd = [dis.calc(grays[i + 1], grays[i], None) for i in range(n - 1)]  # i+1 -> i

    areas = np.array([float((p >= 0.5).mean()) for p in probs])
    out: list[np.ndarray] = []

    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        local_med = float(np.median(areas[lo:hi]))
        trust_self = local_med <= 1e-6 or abs(areas[i] - local_med) <= area_gate * local_med

        acc = np.zeros_like(probs[i], dtype=np.float32)
        wsum = 0.0
        if trust_self:
            acc += probs[i]
            wsum += 1.0

        flow = np.zeros((*probs[i].shape, 2), np.float32)  # i -> i (identity)
        for j in range(i - 1, lo - 1, -1):
            flow = _compose(flow, step_bwd[j])  # step_bwd[j]: (j+1) -> j
            w = decay ** (i - j)
            acc += w * _warp(probs[j], flow)
            wsum += w

        flow = np.zeros((*probs[i].shape, 2), np.float32)
        for j in range(i + 1, hi):
            flow = _compose(flow, step_fwd[j - 1])  # step_fwd[j-1]: (j-1) -> j
            w = decay ** (j - i)
            acc += w * _warp(probs[j], flow)
            wsum += w

        out.append(acc / max(wsum, 1e-6))

    return out


# --------------------------------------------------------------------------
# contact sheet
# --------------------------------------------------------------------------

def _contact_sheet(bgrs: list[np.ndarray], masks: list[np.ndarray], cols: int = 10, thumb_w: int = 240) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    tiles = []
    for bgr, mask in zip(bgrs, masks):
        h, w = bgr.shape[:2]
        th = max(1, round(h * thumb_w / w))
        tile = cv2.resize(bgr, (thumb_w, th))
        m = cv2.resize(mask, (thumb_w, th), interpolation=cv2.INTER_NEAREST)
        bg = m < 128
        tile[bg] = (tile[bg] * 0.35).astype(np.uint8)
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel) > 0
        tile[edge] = (0, 255, 0)
        tiles.append(tile)

    row_h = max(t.shape[0] for t in tiles)
    blank = np.zeros((row_h, thumb_w, 3), np.uint8)
    padded = [np.vstack([t, blank[t.shape[0]:]]) if t.shape[0] < row_h else t for t in tiles]
    while len(padded) % cols:
        padded.append(blank)
    rows = [np.hstack(padded[r:r + cols]) for r in range(0, len(padded), cols)]
    return np.vstack(rows)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_masking(cfg: MaskConfig) -> dict:
    images = sorted(p for p in Path(cfg.image_dir).iterdir() if p.suffix in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"no images in {cfg.image_dir}")

    masker = _clipseg_masker(cfg.prompt) if cfg.prompt else _saliency_masker(cfg.model, cfg.alpha_matting)
    source = f"clipseg[{cfg.prompt!r}]" if cfg.prompt else f"rembg[{cfg.model}]"

    bgrs, raw_probs = [], []
    for path in images:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"could not read image: {path}")
        bgrs.append(bgr)
        raw_probs.append(masker(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    full_h, full_w = bgrs[0].shape[:2]

    if cfg.temporal and len(images) >= 3:
        scale = min(1.0, cfg.flow_max_width / full_w)
        fw, fh = max(1, round(full_w * scale)), max(1, round(full_h * scale))
        grays = [cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (fw, fh)) for b in bgrs]
        small = [cv2.resize(p, (fw, fh), interpolation=cv2.INTER_LINEAR) for p in raw_probs]
        stabilized = _stabilize(small, grays, cfg.temporal_window, cfg.temporal_area_gate)
        probs = [cv2.resize(p, (full_w, full_h), interpolation=cv2.INTER_LINEAR) for p in stabilized]
    else:
        probs = raw_probs

    mask_dir = Path(cfg.output_dir) / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = None
    if cfg.composite:
        comp_dir = Path(cfg.output_dir) / "images_masked"
        comp_dir.mkdir(parents=True, exist_ok=True)
    fill = 255 if cfg.composite == "white" else 0

    areas, suspicious, out_masks = [], [], []
    for path, bgr, prob in zip(images, bgrs, probs):
        binary = _clean_binary(prob, cfg.threshold)
        mask = (binary.astype(np.uint8) * 255)
        if cfg.feather > 0:
            k = cfg.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        if cfg.invert:
            mask = 255 - mask
        out_masks.append(mask)

        area_frac = float((mask > 127).mean())
        areas.append(area_frac)
        if area_frac < cfg.min_area_frac:
            suspicious.append(path.name)

        out_name = f"{path.name}.png" if cfg.colmap_naming else f"{path.stem}.png"
        cv2.imwrite(str(mask_dir / out_name), mask)
        if comp_dir is not None:
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            out = (bgr.astype(np.float32) * alpha + fill * (1.0 - alpha)).astype(np.uint8)
            cv2.imwrite(str(comp_dir / path.name), out)

    if cfg.contact_sheet:
        sheet_path = Path(cfg.output_dir) / "contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), _contact_sheet(bgrs, out_masks))

    areas = np.array(areas)
    # per-frame jump in kept area -- a stability proxy (lower is steadier)
    jitter = float(np.abs(np.diff(areas)).mean()) if len(areas) > 1 else 0.0
    result = {
        "num_images": len(images),
        "source": source,
        "temporal": bool(cfg.temporal and len(images) >= 3),
        "mean_foreground_frac": float(areas.mean()),
        "min_foreground_frac": float(areas.min()),
        "area_jitter": jitter,
        "suspicious_images": suspicious,
        "mask_dir": str(mask_dir),
    }
    print(f"[mask] {len(images)} images via {source} -> {mask_dir}")
    print(f"[mask] foreground {areas.mean():.1%} mean / {areas.min():.1%} min, "
          f"frame-to-frame area jitter {jitter:.1%}"
          f"{' (temporal on)' if result['temporal'] else ''}")
    if suspicious:
        print(f"[mask] {len(suspicious)} image(s) with a tiny foreground -- check these: "
              f"{', '.join(suspicious[:8])}{' ...' if len(suspicious) > 8 else ''}")
    if comp_dir is not None:
        print(f"[mask] composited images -> {comp_dir}")
    if cfg.contact_sheet:
        print(f"[mask] contact sheet -> {Path(cfg.output_dir) / 'contact_sheet.jpg'}")
    return result
