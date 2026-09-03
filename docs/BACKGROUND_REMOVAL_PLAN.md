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

A per-image binary mask `mask/<name>.png` (255 = object, 0 = background) is
consumed at three points, in increasing order of impact:

1. **COLMAP feature extraction** — `--ImageReader.mask_path`. Keypoints are
   only detected inside the mask, so SfM poses are solved from the object +
   nothing else. *Caveat:* a moving object gives COLMAP inconsistent geometry;
   for regime B you mask the object OUT for SfM (solve poses from the static
   world) and mask it IN for training. Two different masks, opposite polarity.
2. **SfM point-cloud seeding** — drop `points3D` whose track is entirely
   outside the object masks before `init_gaussians`, so no Gaussians start
   life on the background.
3. **Training loss** — apply the mask to the L1/SSIM loss (and composite the
   masked-out region to a constant colour) so gradients never ask a Gaussian
   to reconstruct the background. This is what actually keeps the background
   out of the final splat.

Only (3) is strictly required for a clean result; (1) and (2) improve pose
quality and reduce the floaters (3) then has to prune.

---

## Regime A — appearance-based (implemented as `gsplat-pipeline mask`, v0)

Per-frame salient-object segmentation. No temporal or multi-view reasoning —
each image is segmented independently by "what is the subject of this photo".

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

### v0 → v1 hardening (still appearance-based)
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

1. **v0 (done):** `gsplat-pipeline mask` — BiRefNet per-frame. Validate on
   `car-walkaround`. Good enough for regime-A object splats today.
2. **v1:** wire masks into the pipeline — `--mask-dir` on `train` (loss +
   composite), mask-aware `points3D` filtering in `load_scene`, and
   `--ImageReader.mask_path` in the SfM runner. This is the high-value work
   and is independent of how the masks were produced.
3. **v1.5:** add SAM 2 video propagation as a second `mask` backend
   (`--model sam2`, seed frame + click or auto-box) for stability on ordered
   captures.
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
