# How this was built

This repo reimplements the same COLMAP -> Gaussian Splatting -> viewer
pipeline as `nerfstudio-baseline` (a vendored copy of nerfstudio), but
without vendoring nerfstudio. This is a record of what was taken from where,
for anyone auditing how much is "original" vs "borrowed."

## From nerfstudio (reference only, no code copied)

nerfstudio's `splatfacto` model
(`nerfstudio/models/splatfacto.py`) and `ColmapDataParser`
(`nerfstudio/data/dataparsers/colmap_dataparser.py`) were read in full and
used as the spec for *behavior*, not as source to copy from:

- **All the numeric defaults** in `train.py`'s `TrainConfig` --
  `max_steps=30000`, `warmup_length=500`, `refine_every=100`,
  `reset_alpha_every=30`, `cull_alpha_thresh=0.1`, `cull_scale_thresh=0.5`,
  `densify_grad_thresh=0.0008`, `stop_split_at=15000`, `ssim_lambda=0.2`,
  `sh_degree_interval=1000`, `sh_degree=3` -- are `splatfacto`'s own
  defaults, copied by value so results land in the same range on the same
  data.
- **The eval/train split convention** (every 8th image by filename is
  held out) matches `ColmapDataParser`'s `eval_mode="interval"` default.
- **The insight that camera conventions need to match**: nerfstudio stores
  poses in an OpenGL-derived convention and its `get_viewmat()` has to flip
  axes before calling gsplat's rasterizer. Reading that code is what made it
  obvious this pipeline could skip that step entirely by keeping COLMAP's
  own convention (OpenCV: +Y down, +Z forward) all the way through --
  `train.py`'s `_viewmat_from_camtoworld()` is a bare matrix inverse, and
  `viewer.py` relies on `viser`'s camera already using the same convention.
- **Reading `SETUP-PLAN.md`/`PIPELINE.md`** (this repo's sibling docs) is
  where the COLMAP CLI flag-rename gotcha (`--SiftExtraction.use_gpu` ->
  `--FeatureExtraction.use_gpu`) and the "gsplat JIT-compiles its CUDA
  kernels on first use, not at install time" behavior came from.

No nerfstudio code is imported or copied. There's no `Trainer`, `Pipeline`,
`DataManager`, or plugin/config-registry framework here -- `train.py` is one
function.

## From `gsplat` directly (code + design both reused)

- `gsplat.strategy.DefaultStrategy` (densification: splitting, duplicating,
  and pruning gaussians during training) is used **as-is**, not
  reimplemented -- it ships as part of the `gsplat` package itself and is
  the same object nerfstudio's `splatfacto` model uses.
- `gsplat.rendering.rasterization()` is the rasterizer for both training and
  the viewer.
- `model.py`'s gaussian initialization (seed from SfM points, per-gaussian
  scale from mean 4-nearest-neighbor distance, RGB->SH conversion, identity
  quaternion init, per-parameter Adam optimizers with per-parameter learning
  rates, exponential LR decay on `means` down to 1% of its initial value)
  follows the same recipe as gsplat's own
  [`examples/simple_trainer.py`](https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py)
  (`create_splats_with_optimizers`) -- read for reference, then written
  fresh here without gsplat's example-script extras (appearance
  optimization, bilateral grids, pose optimization, MCMC strategy, PPISP,
  depth loss, DDP, compression, ...).
- `colmap/dataset.py`'s approach -- undistort once up front so the
  rasterizer only ever sees ideal pinhole cameras, and the general shape of
  what a "COLMAP scene" data structure needs to carry (per-image K,
  camera-to-world, size, plus the sparse point cloud) -- follows gsplat's
  own [`examples/datasets/colmap.py`](https://github.com/nerfstudio-project/gsplat/blob/main/examples/datasets/colmap.py)
  `Parser`/`Dataset` split. That file uses `pycolmap` to read the sparse
  model; this repo doesn't (see below).

## Original to this repo

- `colmap/binary.py` -- a from-scratch reader for COLMAP's
  `cameras.bin`/`images.bin`/`points3D.bin` binary format (per the format
  spec at https://colmap.github.io/format.html), so there's no dependency on
  `pycolmap` (whose wheel bundles its own COLMAP build, which can silently
  diverge from whatever `colmap` binary actually produced the model on
  disk -- the exact kind of version-mismatch bug `SETUP-PLAN.md` documents
  happening with `libfaiss`).
- `colmap/runner.py` -- a lean subprocess wrapper around the `colmap` CLI
  (feature extraction -> matching -> mapping). No video/frame-extraction,
  mask handling, or vocab-tree matching support.
- `colmap/dataset.py`'s downscale-factor auto-detection (mirrors
  nerfstudio's `MAX_AUTO_RESOLUTION=1600` heuristic, independently
  reimplemented since it wasn't in gsplat's example loader).
- `viewer.py` -- a `viser`-based interactive viewer, written directly
  against `viser`'s API rather than through nerfstudio's `ViewerState`
  abstraction or gsplat's own `nerfview` companion package (not used here,
  to keep the dependency list smaller).
- `io.py`, `cli.py`, `eval.py`, and all the glue holding the above together.

## Deliberately left out (vs. nerfstudio)

- LPIPS in eval (PSNR/SSIM only) -- avoids a pretrained-network-weights
  download for a metric that isn't needed to sanity-check a run.
- Camera pose optimization, bilateral grid appearance correction, MCMC
  densification strategy, scale regularization, tiling, masks/depth
  supervision -- all present in `splatfacto` behind config flags, none
  used by the reference `bonsai` mission this pipeline was built to match,
  so none are here.
- Any NeRF method that isn't Gaussian Splatting.
