# gsplat-pipeline

A minimal COLMAP -> Gaussian Splatting -> viewer pipeline, built directly on
[gsplat](https://github.com/nerfstudio-project/gsplat) with no other
framework in between.

This exists as a lightweight sibling to a larger vendored-nerfstudio setup:
same three-stage flow (structure-from-motion, then splat training, then an
interactive viewer), same core algorithm and defaults, but as ~1,000 lines of
own code instead of a full multi-method NeRF framework. If you don't need
nerfstudio's other dataparsers, NeRF models, or exporters, this is the same
pipeline without carrying all of that.

## Pipeline

```
photos/ --[COLMAP]--> sparse model --[gsplat]--> trained scene --[viser]--> interactive viewer
         (cameras.bin,                (checkpoint.pt,
          images.bin,                  point_cloud.ply)
          points3D.bin)
```

1. **SfM** (`gsplat-pipeline sfm`) -- shells out to the `colmap` CLI
   (feature extraction, matching, incremental mapping) to recover camera
   poses and a sparse point cloud from a folder of photos.
2. **Train** (`gsplat-pipeline train`) -- initializes one 3D Gaussian per SfM
   point and optimizes position/scale/rotation/opacity/color (spherical
   harmonics) against the input photos, using `gsplat`'s CUDA rasterizer for
   the forward/backward pass and its `DefaultStrategy` for adaptive
   densification (splitting, duplicating, and pruning gaussians during
   training).
3. **View** (`gsplat-pipeline view`) -- loads a checkpoint and serves an
   interactive 3D viewer (pan/orbit/zoom) over `viser`, re-rendering with
   `gsplat` on every camera move.

`gsplat-pipeline eval` renders the held-out views (every 8th image, matching
the Mip-NeRF 360 convention) and reports PSNR/SSIM against ground truth.

## Why this exists / what's deliberately left out

The reference setup this is distilled from vendors all of nerfstudio to run
one specific pipeline (COLMAP -> `splatfacto` -> `ns-viewer`). That means
carrying nerfstudio's other NeRF methods (instant-ngp, nerfacto, ...),
dataparsers for a dozen dataset formats, the exporter framework, the plugin
system, etc. -- none of which the `colmap -> gsplat -> viewer` flow actually
uses. This repo keeps the flow and drops everything else:

- No NeRF methods -- Gaussian Splatting only.
- No dataparser framework -- one COLMAP loader (`colmap/dataset.py`), ~150
  lines, using a hand-written binary-format reader instead of `pycolmap` (so
  there's no ABI coupling to whatever COLMAP build produced the model).
- No training-framework abstraction (`Trainer`/`Pipeline`/`DataManager`) --
  `train.py` is one function with a plain training loop.
- Densification is *not* reimplemented -- it's delegated to
  `gsplat.strategy.DefaultStrategy`, which ships as part of `gsplat` itself
  and is the same strategy nerfstudio's `splatfacto` model uses.
- No CI, docs site, or plugin system.

Same defaults as `splatfacto` where it matters (30k steps, warmup/refine
schedule, loss weights, SH degree schedule, eval-every-8th-image split), so
results should land in the same PSNR range on the same data.

## Install

Requires a CUDA GPU (gsplat JIT-compiles its rasterization kernels against
whatever torch/CUDA build is active the first time you train -- expect a
multi-minute one-time compile on first use) and the `colmap` CLI on `PATH`.

```bash
uv sync
```

or with plain pip:

```bash
pip install -e .
```

COLMAP itself isn't a Python package -- install it via your system package
manager or (recommended, version-pinned) via micromamba, same as the
[Dockerfile](Dockerfile). Use a CUDA-enabled build so SfM runs on the GPU;
pick one whose CUDA floor is at or below your driver's CUDA version
(`conda-forge::colmap=3.10=gpu*` needs `__cuda>=12.0`, 3.11+ GPU builds need
12.6-12.9):

```bash
micromamba create -y -p ./colmap-env -c conda-forge "colmap=3.10=gpu*" "openimageio=3.1.*"
export PATH="$(pwd)/colmap-env/bin:$PATH"
export QT_QPA_PLATFORM=offscreen   # COLMAP CLI aborts on a headless host without this
```

(A CPU-only `colmap` works too -- pass `gsplat-pipeline sfm --no-use-gpu` --
but its SIFT matcher is far slower.)

## Usage

```bash
# 1. Structure from motion: photos/ -> sparse COLMAP model
gsplat-pipeline sfm --image-dir photos/ --output-dir data/scene

# 2. Train: sparse model + photos -> a Gaussian Splatting checkpoint
gsplat-pipeline train --data-dir data/scene --output-dir outputs/scene

# 3. Evaluate on held-out views
gsplat-pipeline eval --data-dir data/scene \
    --checkpoint outputs/scene/checkpoints/final.pt --output-dir outputs/scene

# 4. Interactive viewer
gsplat-pipeline view --checkpoint outputs/scene/checkpoints/final.pt
# then open http://localhost:7007
```

Already have a COLMAP model from elsewhere (e.g. the Mip-NeRF 360 dataset,
which ships its sparse model at `<scene>/sparse/0` rather than the
`sfm` command's default `<output>/sparse/0`)? Skip step 1 and point `train`
at it directly -- `--data-dir` auto-detects `sparse/0`, `colmap/sparse/0`, or
you can pass `--colmap-path` explicitly.

`scripts/run_pipeline.sh` wraps steps 1-3 into one command for a folder of
photos: `IMAGE_DIR=photos/ scripts/run_pipeline.sh`.

## Project layout

```
src/gsplat_pipeline/
├── cli.py              entry point (sfm / train / eval / view subcommands)
├── colmap/
│   ├── binary.py        cameras.bin / images.bin / points3D.bin reader
│   ├── runner.py         subprocess wrapper around the `colmap` CLI
│   └── dataset.py        COLMAP model + images -> undistorted (pose, K, image) samples
├── model.py             Gaussian parameter init + per-parameter optimizers
├── train.py             the training loop
├── eval.py              PSNR/SSIM + held-out-view renders
├── viewer.py             viser-based interactive viewer
└── io.py                checkpoint save/load + .ply export
```

## License

Apache 2.0, matching `gsplat` and nerfstudio upstream.
