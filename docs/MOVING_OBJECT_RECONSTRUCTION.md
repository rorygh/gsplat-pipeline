# Reconstructing a moving rigid object — datasets & techniques

Notes for the eventual capture regime: **one rigid object moves, the world is
static, the camera may also move** (a car driving past; a handheld orbit of a
car on a turntable). This is a different problem from the current parked-car
walkaround and needs motion handling, not just background masking.

The key prior: **rigidity collapses the problem**. A rigid object is 6 DOF per
frame — one `SE(3)` pose — versus thousands of DOF for a deformable object. If
you know it's rigid, parameterise it as rigid; do not reach for a general
deformation field.

---

## Sample datasets

### Kubric MOVi-D / MOVi-E — primary pick (synthetic, exact fit, free)
Procedurally-generated 2 s rigid-body simulations. MOVi-D: static camera,
~10–20 objects of which only a few move. MOVi-E: **camera moves** in a random
direction — the closest match to "moving rigid objects + moving camera +
mostly-static scene". Per-frame ground truth: instance segmentation, depth,
optical flow, surface normals, **per-object 3D pose + velocity**, 2D/3D
bounding boxes, camera intrinsics/extrinsics, collision events.
- Pre-rendered: `gs://kubric-public/tfds` (via `tensorflow_datasets`,
  `movi_e/256x256`). ~9.75k train clips, 24 frames each, 12 fps.
- Generator: <https://github.com/google-research/kubric> — render your own
  with 1 slowly-moving object + realistic HDRI background to mimic the target
  capture.
- Paper: <https://arxiv.org/pdf/2203.03570>

### Street Gaussians' Waymo / KITTI splits — real-world rigid-object-in-motion
Moving vehicles (rigid) against a static world, moving ego camera, with
per-object tracker poses provided. Directly the "rigid object moves, camera
moves, background static" case and the de-facto benchmark for the
scene-graph methods below.
- <https://github.com/zju3dv/street_gaussians> (data prep scripts + splits)
- OmniRe uses the same Waymo NOTR split: <https://ziyc.github.io/omnire/>
- **MV2** (2026): synchronized car/scooter/drone captures of the same
  dynamic urban scene — built for large-viewpoint NVS with moving vehicles.
  <https://arxiv.org/abs/2608.12442>

### STaR dataset — minimal single-rigid-object reference
Synthetic + real multi-view rig, exactly one rigid object moving in an
otherwise static scene. Small, but it is the canonical "single `SE(3)` per
frame" setup. <https://wentaoyuan.github.io/star/>

### Casual monocular (object + camera both move, one handheld video)
- **DyCheck / iPhone dataset** — 14 handheld sequences, 7 with two static
  eval cameras at a wide baseline. Some sequences are a single moving object.
  <https://hangg7.com/dycheck/>
- **NVIDIA Dynamic Scene Dataset** (Yoon et al.) — 8 scenes, moving
  subject + 12-camera rig; mostly people, some rigid props.
- Objectron — object-centric videos with 3D boxes, no NVS ground truth;
  useful for pose supervision only.

---

## Techniques, by how much annotation / prior they need

### 1. Segment-then-reconstruct (what our plan's "regime B" builds toward)
Mask the moving object per frame (motion segmentation — see
[BACKGROUND_REMOVAL_PLAN.md](BACKGROUND_REMOVAL_PLAN.md) §B), then:
- Run SfM on the **masked-out** frames → clean camera poses + static point
  cloud from the background alone (the moving object otherwise corrupts BA).
- Train a static 3DGS for the background.
- Reconstruct the object separately: if the camera moves enough relative to
  the object, its masked pixels across frames are effectively a multi-view
  set in the *object's* frame — feed them (with per-frame object pose from a
  tracker, or bundle-adjusted) into a second 3DGS.
- Composite both at render time.

Robust, uses tools we already have. Weak when the object is textureless (few
features land on it) or the relative object–camera motion is small.

### 2. Compositional scene graph — the driving-scene answer
Two or more Gaussian sets: **world frame** + one **object-local frame** per
rigid mover. Each object carries a per-frame `SE(3)` pose; a Gaussian at
object-local `x` renders at `R_t x + t_t` in world at time `t`. Render the
union, photometric loss, backprop into Gaussians *and* (optionally) the poses.
- **Neural Scene Graphs** (Ost et al., CVPR 2021) — original formulation,
  NeRF nodes.
- **Street Gaussians** — per-object point cloud + optimisable tracked poses +
  a **4D spherical-harmonics** appearance model (time-varying colour for
  headlights / changing reflections). <https://arxiv.org/pdf/2401.01339>
- **OmniRe** — "Gaussian Scene Graph": separate node types for sky,
  background, rigid objects (Gaussians fixed in object frame), non-rigid
  actors. <https://arxiv.org/html/2408.16760>
- **Panoptic Neural Fields**, **DrivingGaussian**, **PVG**, **DeSiRe-GS** —
  variants.

Best when you can get object poses (a 3D tracker, or 2D box + PnP, or
optimise from an init). Handles the "camera also moving" case for free since
everything is in world frame.

### 3. Joint self-supervised static/dynamic factorisation (no masks, no poses)
Optimise the decomposition *and* the motion together from raw video.
- **STaR** (Yuan et al., CVPR 2021) — two radiance fields (static + dynamic) +
  one rigid `SE(3)` per frame, jointly optimised to reconstruct multi-view
  video. Elegant; assumes exactly one rigid mover; optimisation is fragile
  and needs a static-scene warm-up.
- **D²NeRF**, **NeuralDiff** — self-supervised static/transient splitting
  (transient branch is not rigidity-constrained).

### 4. Rigidity-regularised dynamic Gaussians (per-Gaussian motion + priors)
Let every Gaussian move/rotate over time; regularise so neighbours move
together as rigid clusters.
- **Dynamic 3D Gaussians** (Luiten et al., 3DV 2024) — persistent
  colour/opacity/size + three losses: **local rigidity** (neighbour offsets
  are preserved under each Gaussian's local rotation), **local
  rotation-similarity**, **long-term local isometry**. Gives 6-DOF tracking
  of every scene element as a by-product. Needs multi-view video.
  <https://dynamic3dgaussians.github.io/>
- **RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video**
  (2026) — three primitive types (static / rigid = long-term low-frequency /
  transient = short-term high-frequency), an object-wise dynamic mask drives
  the split, rigid primitives can *become* transient over time, all
  supervised by dense **scene flow**. Closest recent work to our target from
  a single video. <https://arxiv.org/abs/2605.23672>

### 5. Casual monocular 4D (one handheld video, object + camera both move)
Foundation-model-driven; the general-purpose version of the problem.
- **Shape of Motion** (2024) — represents scene motion as a small set of
  **`SE(3)` motion bases** shared across the scene; each Gaussian's
  trajectory is a low-rank combination. For a *single rigid object* this
  collapses to ~one basis = exactly rigid motion, so it degrades gracefully
  to the rigid case. Priors: mono depth + long-range 2D tracks.
  <https://shape-of-motion.github.io/>
- **MoSca** (2024) — a **4D Motion Scaffold** graph of trajectory nodes with
  an **as-rigid-as-possible (ARAP)** regulariser; fuses mono depth + tracks +
  SAM2 masks. <https://arxiv.org/abs/2405.17421>
- **Dynamic Gaussian Marbles** (SIGGRAPH Asia 2024), **Gaussian Sequences**
  (2026) — related casual-video 4DGS.

### 6. SfM / pose estimation under dynamics (upstream of all the above)
Standard COLMAP breaks when a large moving object is in frame.
- Mask dynamic content before COLMAP (**RoDynRF**, robust BA).
- **ParticleSfM**, **CasualSAM** — jointly estimate motion masks + camera
  poses + depth from the video.
- Or: SfM on background-masked frames (technique 1).

---

## Shared machinery (motion cues every method leans on)

| Cue | Models |
|---|---|
| Optical flow | RAFT, SEA-RAFT |
| Long-range 2D point tracks | CoTracker3, BootsTAPIR, TAPIR |
| Monocular depth | Depth-Anything V2, UniDepth, MoGe |
| Mask cleanup / propagation | SAM 2, Track-Anything |
| Motion segmentation | SegAnyMo (tracks + depth → dynamic-track classifier) |
| Scene flow | lift flow + depth to 3D, or GS-based (RiGS) |

---

## Recommendation for this pipeline

1. **Prototype on Kubric MOVi-E** — it has everything (GT object poses,
   masks, flow, depth), so each component can be validated against ground
   truth before touching real captures.
2. **Start with technique 1 + technique 2**: motion-mask the object, SfM +
   static 3DGS on the background, then a second object-frame Gaussian set with
   per-frame poses (init from a tracker, refine by optimisation). This reuses
   the existing SfM → 3DGS spine — the object is "just another 3DGS in a
   moving coordinate frame".
3. Add **Dynamic-3DGS local-rigidity losses** to the object Gaussians if
   optimising poses from scratch — they keep the object from smearing.
4. Only reach for full casual-4D (Shape of Motion / MoSca) if captures end up
   genuinely unconstrained (object *and* camera moving freely, no poses).
5. Real-world validation set: **Street Gaussians' Waymo split** (rigid movers,
   provided poses) before shooting bespoke moving-car footage.

## Sources

- Kubric / MOVi: <https://github.com/google-research/kubric/blob/main/challenges/movi/README.md> · <https://arxiv.org/pdf/2203.03570>
- STaR: <https://wentaoyuan.github.io/star/> · <https://arxiv.org/abs/2101.01602>
- Neural Scene Graphs: <https://light.princeton.edu/publication/neural-scene-graphs/>
- Street Gaussians: <https://arxiv.org/pdf/2401.01339> · <https://github.com/zju3dv/street_gaussians>
- OmniRe: <https://ziyc.github.io/omnire/>
- Dynamic 3D Gaussians: <https://dynamic3dgaussians.github.io/> · <https://arxiv.org/abs/2308.09713>
- RiGS: <https://arxiv.org/abs/2605.23672>
- Shape of Motion: <https://shape-of-motion.github.io/> · <https://arxiv.org/pdf/2407.13764>
- MoSca: <https://arxiv.org/abs/2405.17421>
- DyCheck iPhone dataset: <https://hangg7.com/dycheck/>
- MV2: <https://arxiv.org/abs/2608.12442>
