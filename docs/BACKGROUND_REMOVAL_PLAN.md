# Background removal — plan

Goal: train a Gaussian Splatting scene of **one object** with the background
excluded, so the splat contains only the object (clean export, no floaters
from parked cars / houses / foliage, correct bounding box).

Two regimes, addressed in order:

| Regime | Object | Background | Cue that separates them |
|---|---|---|---|
| **A — now** | stationary | stationary | appearance / saliency only |
| **B — later** | **moving** | stationary | motion (object moves against a fixed world) |

The current `car-walkaround` capture is regime A (car parked, camera orbits).
The eventual captures are regime B (car drives, camera fixed or moving).

---

## What the mask feeds into

A per-image binary mask `mask/<stem>.png` (255 = object, 0 = background) is
consumed at four points. **All four are implemented** (`train --mask-dir`,
`sfm --mask-path`):

1. **COLMAP feature extraction** — `--ImageReader.mask_path` (`run_sfm(...,
   mask_path=)`). Keypoints only inside the mask → poses from the object.
   *Caveat:* a moving object gives COLMAP inconsistent geometry; there you
   mask the object **OUT** for SfM (`mask --invert --colmap-naming`, poses
   from the static world) and **IN** for training. Opposite polarity.
2. **SfM point-cloud seeding** — `_points_in_masks` in `colmap/dataset.py`:
   a point survives only if it projects inside the (dilated) mask in ≥1 of
   its observing views. Everything else — the whole background — is dropped
   before `init_gaussians`.
3. **Training loss** — `train.py`: composite pred and GT onto the same colour
   outside the mask (so windowed SSIM still works), L1+SSIM on that, **plus**
   an `alpha_mask_lambda · MSE(rendered_alpha, mask)` term so no Gaussian
   grows off-object.
4. **Final prune** — `gaussians_in_masks`: after training, drop any Gaussian
   whose centre never projects into a training mask. Guarantees the saved
   checkpoint/PLY is the object only.

(3)+(4) are what make the result object-only; (1)+(2) reduce the work they
have to do. Measured on `tissue-paper`: SfM seed 6853 → 1757 points on the
object; see `reports/` for the masked-vs-unmasked comparison.

---

## Regime A — appearance-based (implemented as `gsplat-pipeline mask`)

**Status (v0.2):** two raw-mask backends + a temporal-stabilisation pass.

- **Saliency** (default): rembg / BiRefNet, per-frame. GPU via
  `onnxruntime-gpu` (`_preload_cuda_libs` reuses torch's bundled cuDNN/cuBLAS).
- **Text prompt** (`--prompt "a car"`): CLIPSeg (`transformers`) maps the
  phrase to a soft mask; thresholded (`--threshold`, default 0.5) + cleaned.
  Disambiguates scenes where saliency latches onto the wrong subject.
- **Temporal stabilisation** (`--temporal`, on by default,
  `masking._stabilize`): DIS optical flow between consecutive frames; each
  soft mask is re-estimated as a flow-warped vote over a ±`temporal_window`
  window, and any frame whose area jumps past `temporal_area_gate` of the
  local median is rebuilt from neighbours. Measured on `car-walkaround`:
  frame-to-frame area jitter 12.1% → 7.2% (saliency) and → 3.6% (prompt).
- **Cleanup** (`_clean_binary`): threshold → largest connected component →
  fill holes → feather.
- **Outputs**: `mask/<stem>.png`, `images_masked/`, `contact_sheet.jpg`
  (thumbnails with the mask outlined -- for eyeballing stability), plus an
  `area_jitter` number in the returned stats. Default dir `./masks/`
  (gitignored).

What's still missing from v0.2: multi-view hull consistency, SAM 2 video
propagation, and -- the actually-important part -- wiring the masks into
training (loss + point filtering + SfM). See below.

### Per-frame backend notes

- **Model:** BiRefNet (`birefnet-general` via `rembg`) — high-resolution
  dichotomous segmentation, automatic (no prompt), commercially licensed
  (MIT). Fallbacks: `isnet-general-use` (lighter), `u2net` (original rembg).
- **When it works:** the object dominates each frame and is roughly centred
  (walkaround, turntable). Our contact sheet shows the Elantra filling
  40–70 % of frame — good candidate.
- **When it breaks:** object small in frame, multiple plausible "subjects"
  (two cars), object colour ≈ background, thin structure (antennas, wipers).
- **Failure signal:** `mask` already flags frames whose foreground is
  < 1 % of the image (`suspicious_images` in the JSON).

### v1 hardening (still appearance-based)
- **Temporal propagation via optical flow** — *done in v0.2* (`_stabilize`).
- **Text-prompted segmentation** — *done in v0.2* (CLIPSeg `--prompt`). A
  heavier GroundingDINO→SAM 2 path is still an option for hard cases.
- **Multi-view consistency check.** Un-project each mask into 3D using the
  COLMAP poses + a coarse depth (median SfM-point depth in the mask), take
  the intersection of the visual hull across all views, re-project → a
  consistent mask per frame. Kills per-frame flicker and latch-on errors.
- **Prompted segmentation** for ambiguous scenes: GroundingDINO text prompt
  (`"car"`) → box → SAM 2 mask. One prompt covers the whole set.
- **Temporal propagation:** SAM 2 video mode — mask the object once on the
  sharpest frame, propagate through the ordered sequence. Cheap, very stable
  for a walkaround where frames are ordered.

---

## Regime B — motion-based (object moves, world static)

Now there *is* a hard cue: over the sequence, the object's pixels move in a
way the static background doesn't. Options, cheapest first:

### B1. Reconstruction residual (reuses what we already run)
1. Run SfM on the **whole** scene. The static background triangulates
   cleanly; the moving object's features either fail to triangulate or land
   with large reprojection error / short tracks.
2. Mark as "moving" any `points3D` with high error or track length ≤ 2, plus
   2D keypoints that never triangulated.
3. Re-project those into each frame, densify to a mask (superpixel or SAM 2
   seeded by the moving points).
- **Pros:** no new models, no optical flow; falls straight out of the COLMAP
  stage; the `colmap-report` reprojection-error histogram already surfaces
  the signal.
- **Cons:** coarse; struggles if the object is textureless (our black car
  body) so few points land on it anyway; needs enough camera motion that the
  background parallax is distinguishable from object motion.

### B2. Optical-flow / epipolar residual
Estimate camera motion between consecutive frames (from SfM poses), predict
the flow the static scene *should* produce, subtract measured flow (RAFT).
Pixels with large residual flow = moving object. Threshold → mask → clean up
with SAM 2.
- **Pros:** dense, works on low-texture objects (flow is regularised).
- **Cons:** needs a flow model (RAFT / SEA-RAFT), tuning; fast camera motion
  or rolling shutter adds noise; degenerate when the object moves along an
  epipolar line.

### B3. Learned motion segmentation (SOTA, heaviest)
**SegAnyMo** (CVPR 2025, `nnanhuang/SegAnyMo`): off-the-shelf 2D point tracks
(CoTracker/BootsTAP) + depth (Depth-Anything / UniDepth) → a motion encoder
classifies each track as dynamic vs static → SAM 2 groups dynamic tracks into
per-object masks. Purpose-built for exactly "segment the moving thing in a
monocular video".
- **Pros:** best quality; handles low texture (tracks are semi-dense and
  regularised); outputs clean SAM 2 masks directly; ordered-video native.
- **Cons:** multi-model pipeline (tracker + depth + their net + SAM 2), ~A6000
  to run comfortably, licence is research-oriented — fine for internal R&D,
  check before shipping. Env: Ubuntu 22.04 / py3.12 / torch 2.4 (matches our
  pod).
- **Related / alternates:** "Segment Any Motion in Videos" is the same work;
  CasualSAM / ParticleSfM jointly optimise motion masks with pose+depth (more
  than we need); classical background subtraction is out (moving camera).

### B4. The clean-plate shortcut
If any capture includes frames of the scene **without** the object (drive the
car out of frame, keep filming), that's a reference background — per-pixel
difference after homography/pose alignment gives a mask almost for free.
Worth building the capture protocol around.

---

## Recommended path

1. **v0.2 (done):** `gsplat-pipeline mask` — saliency **or** CLIPSeg text
   prompt, plus an optical-flow temporal-stabilisation pass and a contact
   sheet. Validated on `car-walkaround` and `tissue-paper`.
2. **v1 (done):** masks wired into the pipeline — `train --mask-dir` (point
   filter + masked loss + alpha supervision + final prune), `eval --mask-dir`
   (metrics on the object region), `sfm --mask-path`, and `mask --invert
   --colmap-naming` for the moving-object SfM polarity.
3. **v1.5 (next):** SAM 2 video propagation as a third `mask` backend for the
   hardest ordered captures; multi-view hull consistency as a backend-agnostic
   refinement (reuses the poses `orientation.py` already needs).
4. **Regime B, first cut:** B1 (reconstruction residual) — small, reuses the
   COLMAP stage, no new deps. Ship it as `mask --mode motion`.
5. **Regime B, quality:** integrate **SegAnyMo** as `mask --mode motion
   --model seganymo` once B1's limits bite (expected: textureless object,
   subtle motion).

### Capture-side asks (cheap, high leverage)
- Grab 5–10 frames of the scene with the object absent (enables B4 and gives
  every method a background prior).
- Keep frames ordered / EXIF timestamps intact (enables SAM 2 video + track
  methods).
- For regime B, prefer *some* camera motion over a locked-off tripod — B1/B2
  need background parallax to distinguish "moving object" from "moving
  camera sees static scene".

---

## References

- BiRefNet — high-res dichotomous segmentation:
  <https://ice-ice-bear.github.io/posts/2026-04-15-birefnet/>
- rembg (BiRefNet / ISNet / U²-Net sessions, MIT):
  <https://github.com/danielgatis/rembg>
- SegAnyMo — Segment Any Motion in Videos (CVPR 2025):
  <https://github.com/nnanhuang/SegAnyMo> · <https://arxiv.org/pdf/2503.22268>
- Segment-and-Track-Anything (SAM + AOT video propagation):
  <https://github.com/z-x-yang/Segment-and-Track-Anything>
- Clean-GS — semantic-mask pruning of a trained 3DGS:
  <https://github.com/smlab-niser/clean-gs>
- MATT-GS — U²-Net background removal + SfM keypoint reduction before 3DGS:
  <https://arxiv.org/pdf/2503.19330>
- Object-Centric 2D Gaussian Splatting — SAM 2 masks + occlusion-aware
  pruning: <https://www.scitepress.org/Papers/2025/133055/133055.pdf>
- On Moving Object Segmentation from Monocular Video with Transformers:
  <https://arxiv.org/pdf/2411.19141>
- 3Dflow — masking for turntable captures (why a non-uniform rotating-object
  background confuses SfM):
  <https://www.3dflow.net/technology/documents/3df-zephyr-tutorials/tutorial-12-masking-for-turntable/>
