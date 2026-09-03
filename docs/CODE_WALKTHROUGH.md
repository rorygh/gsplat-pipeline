# gsplat-pipeline — full code walkthrough

Steps through every module, function, and notable block of the pipeline.
Ordered by the path data takes through the system: CLI dispatch → SfM →
COLMAP model reading → dataset build → Gaussian init → training loop →
checkpoint/PLY I/O → eval → viewer.

File tree under `src/gsplat_pipeline/`:

```
cli.py              subcommand dispatch (sfm / train / eval / view)
colmap/
  binary.py         hand-written reader for COLMAP's *.bin sparse model
  runner.py         subprocess wrapper around the `colmap` CLI
  dataset.py        COLMAP model + images -> undistorted (pose, K, image) samples
model.py            Gaussian parameter init + per-parameter optimizers/scheduler
train.py            the training loop
io.py               checkpoint save/load + .ply export
eval.py             PSNR/SSIM on held-out views + side-by-side renders
viewer.py           viser-based interactive viewer
```

Supporting files: `tests/test_colmap_binary.py`, `tests/test_model.py`,
`scripts/run_pipeline.sh`, `Dockerfile`, `.github/workflows/docker-publish.yml`,
`pyproject.toml`.

---

## 1. `cli.py` — command-line entry point

Defines the `gsplat-pipeline` console script (wired in `pyproject.toml`
`[project.scripts]`). Uses `tyro.extras.SubcommandApp`: each decorated
function becomes a subcommand, and its signature (type hints + defaults)
becomes that subcommand's `--flags` automatically.

### `app = SubcommandApp()`
Module-level registry. The four `@app.command` functions below attach to it.

### `sfm(image_dir, output_dir, matching_method="exhaustive", camera_model="OPENCV")`
- CLI: `gsplat-pipeline sfm --image-dir photos/ --output-dir data/scene`
- `matching_method` is `Literal["exhaustive", "sequential"]` — tyro renders
  this as a constrained choice. Exhaustive compares every image pair (correct
  for unordered photo sets); sequential only compares nearby frames (faster,
  for video).
- `camera_model` is passed straight through to COLMAP's feature extractor.
  `OPENCV` = pinhole + 2 radial + 2 tangential distortion coefficients.
- Body: lazy-imports `run_sfm` (keeps `--help` and other subcommands from
  importing subprocess/pathlib machinery they don't need), calls it, prints
  where the sparse model landed.

### `train(data_dir, output_dir, colmap_path=None, downscale_factor=None, max_steps=30_000, device="cuda")`
- Lazy-imports `TrainConfig` and `train` (as `run_train`) from `train.py`.
- `downscale_factor` translation: the CLI exposes `None` (meaning "not
  specified"), but `TrainConfig` wants the sentinel string `"auto"` for
  auto-detection. Line 49 does that mapping:
  `downscale_factor if downscale_factor is not None else "auto"`.
  So: pass nothing → `"auto"`; pass `--downscale-factor 4` → `4`.
- Builds a `TrainConfig` from the handful of CLI-exposed fields (the rest of
  `TrainConfig`'s ~25 fields keep their dataclass defaults) and calls
  `run_train(cfg)`, which returns the path to `final.pt`.

### `eval_cmd(...)` — registered as `@app.command(name="eval")`
- Named explicitly because `eval` shadows the Python builtin; the function is
  `eval_cmd` but the subcommand is `eval`.
- Same `downscale_factor` `None -> "auto"` mapping as `train`.
- Requires `--checkpoint` (no default). Builds `EvalConfig`, calls
  `evaluate(cfg)` which prints metrics and writes `metrics.json`.

### `view(checkpoint, port=7007, device="cuda")`
- Minimal: builds `ViewerConfig`, calls `run_viewer`. Blocks until Ctrl-C.

### `main()`
`app.cli()` — tyro parses `sys.argv`, picks the subcommand, binds flags to
the function's parameters, calls it. `main` is the `[project.scripts]`
entry point.

---

## 2. `colmap/runner.py` — driving the COLMAP CLI

COLMAP is a C++ binary, not a Python package. This module shells out to it
in three steps: **feature extraction → matching → mapping (bundle
adjustment)**. Output: `<output_dir>/sparse/0/{cameras,images,points3D}.bin`.

### `_run(cmd: list[str])`
Prints the command (so the console log shows exactly what ran) then
`subprocess.run(cmd, check=True)` — `check=True` raises `CalledProcessError`
on a non-zero exit, so a failed COLMAP stage aborts the pipeline loudly
instead of silently continuing to the next stage.

### `_use_gpu_flag(binary: str) -> bool`
Handles a COLMAP CLI breaking change. Older COLMAP: `--SiftExtraction.use_gpu`
and `--SiftMatching.use_gpu`. Newer COLMAP (after 4.0): `--FeatureExtraction.use_gpu`
and `--FeatureMatching.use_gpu`. Rather than hard-coding a version cutoff,
this runs `colmap feature_extractor -h` and greps its help text for
`FeatureExtraction.use_gpu`.
- Note the comment: `colmap -h` alone only lists subcommand *names*, not their
  flags, so it has to invoke the specific subcommand's help.
- Note: COLMAP writes help to **stderr**, hence `.stderr` not `.stdout`.
- Returns `True` if the new flag name is present.

### `run_sfm(image_dir, output_dir, matching_method="exhaustive", use_gpu=True, camera_model="OPENCV", single_camera=True) -> Path`
1. **Guard:** `shutil.which("colmap") is None` → `RuntimeError` with a pointer
   to the README install instructions.
2. **Setup dirs:** creates `output_dir`, `output_dir/sparse`. Database goes to
   `output_dir/database.db` (COLMAP's SQLite store of features + matches).
3. **Pick flag names:** calls `_use_gpu_flag` once, builds
   `extract_gpu_flag` / `match_gpu_flag` strings for the current binary.
4. **Feature extraction** (`_run`): `colmap feature_extractor` with
   `--database_path`, `--image_path`, the GPU flag, `--ImageReader.camera_model`,
   and `--ImageReader.single_camera` (`1` = assume all photos came from one
   physical camera with fixed intrinsics — correct for a single-camera capture,
   and it makes the reconstruction more constrained/stable).
5. **Matching** (`_run`): `exhaustive_matcher` or `sequential_matcher`
   depending on `matching_method`, plus the GPU flag.
6. **Mapping** (`_run`): `colmap mapper` — incremental SfM + bundle
   adjustment. Reads the database, writes reconstruction(s) to
   `output_dir/sparse/`. COLMAP numbers reconstructions `0`, `1`, ... — the
   first (usually only) one is `sparse/0`.
7. **Verify:** if `sparse/0/cameras.bin` doesn't exist, the mapper failed to
   register a model (usually: not enough image overlap) → `RuntimeError`.
8. Returns `Path(output_dir/sparse/0)`.

What it deliberately does NOT do: video frame extraction, mask handling,
vocab-tree matching, image undistortion (that happens later in `dataset.py`),
multi-model merging.

---

## 3. `colmap/binary.py` — reading COLMAP's binary sparse format

A from-scratch `struct`-based reader for the three `.bin` files COLMAP
writes. No `pycolmap` dependency (whose wheel bundles its own COLMAP build
that can silently disagree with the `colmap` binary that produced the model).
Format spec: https://colmap.github.io/format.html

### Module constants
- `CAMERA_MODEL_NUM_PARAMS: dict[int, int]` — maps COLMAP's numeric
  `model_id` to how many `float64` params follow in `cameras.bin`. Only the 5
  models COLMAP's default pipeline actually produces are listed:
  `0 SIMPLE_PINHOLE (3)`, `1 PINHOLE (4)`, `2 SIMPLE_RADIAL (4)`,
  `3 RADIAL (5)`, `4 OPENCV (8)`. This count is needed to know how many bytes
  to read before the next camera record.
- `CAMERA_MODEL_NAMES: dict[int, str]` — same keys → human-readable names.

### `@dataclass Camera`
Fields `id, model (str), width, height, params (np.ndarray)`. The meaning of
`params` depends on `model`.

#### `Camera.as_intrinsics() -> (K, dist)`
Converts COLMAP's model-specific param vector into a standard OpenCV
`(K, dist)` pair that `cv2.initUndistortRectifyMap` can consume:
- `SIMPLE_PINHOLE`: `params = [f, cx, cy]` → `fx = fy = f`, zero distortion.
- `PINHOLE`: `[fx, fy, cx, cy]` → zero distortion.
- `SIMPLE_RADIAL`: `[f, cx, cy, k]` → `dist = [k, 0, 0, 0]`.
- `RADIAL`: `[f, cx, cy, k1, k2]` → `dist = [k1, k2, 0, 0]`.
- `OPENCV`: `[fx, fy, cx, cy, k1, k2, p1, p2]` → `dist = [k1, k2, p1, p2]`.
- Anything else → `ValueError`.
- Builds `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` as float64.
- `dist` follows OpenCV's `[k1, k2, p1, p2]` ordering (radial, radial,
  tangential, tangential).

### `@dataclass Image`
One registered photo: `id`, `qvec` (4-vector `w, x, y, z` — **world-to-camera**
rotation, scalar-first), `tvec` (3-vector world-to-camera translation),
`camera_id` (which `Camera` it used), `name` (filename), `point3D_ids`
(per-2D-keypoint 3D point id, `-1` where the keypoint didn't triangulate).

### `@dataclass Point3D`
One sparse SfM point: `id`, `xyz`, `rgb` (uint8), `error` (reprojection
error), `image_ids` + `point2D_idxs` (the "track" — which images saw this
point and at which keypoint index).

### `qvec2rotmat(qvec) -> 3x3`
Quaternion → rotation matrix. Comment flags the gotcha: COLMAP quaternions
are **scalar-first** (`w, x, y, z`), unlike scipy's scalar-last convention.
The formula is the standard expansion.

### `_read(fid, fmt, endian="<")`
Helper: `struct.calcsize` to get byte width, read exactly that many bytes,
`struct.unpack`. Always little-endian (`<`) — COLMAP's format is
little-endian on disk regardless of host.

### `read_cameras_binary(path) -> dict[int, Camera]`
- Read `Q` (uint64) `num_cameras`.
- Per camera: `iiQQ` = `camera_id (int32), model_id (int32), width (uint64),
  height (uint64)`, then `d * num_params` doubles (count from
  `CAMERA_MODEL_NUM_PARAMS[model_id]`).
- Key the dict by `camera_id`.

### `read_images_binary(path) -> dict[int, Image]`
- Read `Q` `num_images`.
- Per image: `idddddddi` = `image_id (int32), qw, qx, qy, qz, tx, ty, tz
  (7 doubles), camera_id (int32)`.
- Then a **null-terminated filename**: read one `c` (char) at a time until
  `\x00`. (COLMAP stores the name as a C string, not length-prefixed.)
- Then `Q` `num_points2D`, then `num_points2D * (ddq)` =
  `(x, y, point3D_id)` per keypoint. Only `point3D_id` is kept
  (`xys_and_ids[2::3]`), as int64, with `-1` meaning "no 3D point".
- Key by `image_id`.

### `read_points3D_binary(path) -> dict[int, Point3D]`
- Read `Q` `num_points`.
- Per point: `QdddBBBd` = `point3D_id (uint64), x, y, z (3 doubles),
  r, g, b (3 uint8), error (double)`.
- Then `Q` `track_length`, then `track_length * (ii)` = `(image_id,
  point2D_idx)` pairs. Split into `image_ids = track[0::2]`,
  `point2D_idxs = track[1::2]`.
- Key by `point3D_id`.

### `read_model(sparse_dir) -> (cameras, images, points3D)`
Convenience: calls all three readers on
`sparse_dir/{cameras,images,points3D}.bin`, returns the three dicts.

### Test coverage (`tests/test_colmap_binary.py`)
Writes synthetic `.bin` files byte-for-byte in the format above (a 1-camera,
2-image, 2-point model), reads them back, asserts:
- camera model/size/params round-trip,
- both image names present, identity rotation → identity matrix, `tvec` correct,
- point count + RGB round-trip,
- `as_intrinsics()` produces the expected `K` and zero `dist` for PINHOLE.
No COLMAP binary or GPU needed — pure `struct` round-trip.

---

## 4. `colmap/dataset.py` — COLMAP model → training samples

Turns `(cameras, images, points3D)` plus an images folder into:
a `SceneData` bundle, a train/eval index split, and a lazy-loading
`torch.utils.data.Dataset`. **Undistortion happens here, once**, so the
rasterizer only ever sees ideal pinhole cameras.

### `MAX_AUTO_RESOLUTION = 1600`
The auto downscale target: keep the long edge under this. Matches
nerfstudio's `ColmapDataParser`. Full-res phone/DSLR photos (4000+ px) make
each training iteration slow and rarely help splat quality, so the default
downscales them.

### `@dataclass SceneData`
Everything the trainer/eval needs, all NumPy (moved to torch per-sample):
- `image_paths: list[Path]`, `image_names: list[str]` — sorted by filename.
- `camtoworlds: (N,4,4) float64` — **camera-to-world**, OpenCV convention
  (+X right, +Y down, +Z forward). This is the inverse of COLMAP's stored
  world-to-camera.
- `Ks: (N,3,3)` — **undistorted** intrinsics, one per image (post-downscale).
- `widths, heights: (N,) int` — final pixel dims after downscale + undistort.
- `undistort_maps: list[(mapx, mapy) | None]` — per-image `cv2.remap` LUTs,
  or `None` if that camera was already distortion-free.
- `points_xyz: (M,3) float32`, `points_rgb: (M,3) uint8` — the sparse cloud,
  used to seed Gaussians.
- `scene_scale: float` — rough scene radius (see below); scales learning
  rates and densification thresholds.

### `_undistort_image(image, mapx, mapy)`
If `mapx is None`, return the image unchanged. Otherwise
`cv2.remap(image, mapx, mapy, INTER_LINEAR)` — resamples the distorted image
onto the ideal pinhole grid.

### `_pick_auto_downscale_factor(width, height) -> int`
Starting at `factor = 1`, double it while `max(w, h) / factor >
MAX_AUTO_RESOLUTION`. So a 4000px image → factor 4 (→ 1000px); a 1500px
image → factor 1. Always a power of two.

### `load_scene(data_dir, sparse_path=None, images_path=None, downscale_factor="auto") -> SceneData`
The core loader. Steps:

1. **Locate the sparse model.** If `sparse_path` not given, probe
   `data_dir/{sparse/0, colmap/sparse/0, sparse}` for a `cameras.bin`; first
   hit wins. None → `FileNotFoundError`.
2. `cameras, images, points3D = read_model(sparse_path)`.
3. **Resolve the downscale factor.**
   - `"auto"`: `_pick_auto_downscale_factor` on the first camera's size. Then
     a safety check: if factor > 1 but no explicit `images_path` and no
     `data_dir/images_{factor}/` folder exists, print a warning and fall back
     to factor 1. (This pipeline never *generates* downscaled folders — it
     only uses ones already on disk, e.g. as shipped by Mip-NeRF 360.)
   - explicit int: used as-is (`downscale_factor or 1` guards `0`/`None`).
4. **Resolve the images folder.** `images_path = data_dir/images_{factor}` if
   factor > 1 else `data_dir/images`.
5. **Sort images by filename** (`sorted(images.keys(), key=lambda i:
   images[i].name)`) — this ordering is what makes the "every 8th image"
   eval split deterministic and comparable to published results.
6. **Per image, build the calibration:**
   - `K, dist = cam.as_intrinsics()`.
   - `target_w, target_h = cam.width // factor, cam.height // factor`.
   - If downscaling, copy `K` and divide its first two rows by `factor`
     (scales `fx, fy, cx, cy`; the bottom row stays `[0,0,1]`).
   - **Pose:** build `w2c` (4x4) from `qvec2rotmat(im.qvec)` and `im.tvec`,
     then `camtoworlds.append(np.linalg.inv(w2c))`.
   - **Undistortion:** if `dist` is not all-zero,
     `cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)` gives a new
     pinhole `K` with no black border (`alpha=0` crops to all-valid pixels),
     then `cv2.initUndistortRectifyMap` builds the `(mapx, mapy)` LUT. Store
     the map and the *new* `K`. If `dist` is zero: `undistort_maps.append(None)`,
     keep the original `K`.
   - Record `target_w/h`, the image path (`images_path / im.name`), the name.
7. **Stack** the per-image lists into arrays.
8. **Sparse cloud:** `points_xyz` / `points_rgb` from `points3D.values()`.
9. **Scene scale:** `camera_locations = camtoworlds[:, :3, 3]` (all camera
   positions); `scene_center = mean`; `scene_scale = max distance from any
   camera to the center`. A cheap proxy for "how big is this scene" that the
   optimizer uses to normalize step sizes.
10. Return the populated `SceneData`.

### `train_eval_split(num_images, eval_every=8) -> (train_idx, eval_idx)`
`indices = arange(num_images)`. `eval_idx` = indices where `i % 8 == 0`
(0, 8, 16, ...); `train_idx` = the rest. Matches the Mip-NeRF 360 / nerfstudio
`ColmapDataParser` `eval_mode="interval"` convention, so PSNR is comparable to
published numbers. Note: index 0 is always an eval view.

### `class GaussianSplattingDataset(torch.utils.data.Dataset)`
- `__init__(scene, indices)` — holds the `SceneData` and a subset of indices
  (either `train_idx` or `eval_idx`).
- `__len__` — number of indices in this subset.
- `__getitem__(item)`:
  - `idx = self.indices[item]` (map subset position → scene index).
  - Load the image: `cv2.imread` (BGR) → `cv2.cvtColor(..., BGR2RGB)`.
  - `mapx, mapy = scene.undistort_maps[idx] or (None, None)` then
    `_undistort_image`.
  - Return a dict: `image` (float tensor, still 0–255), `K` (float32),
    `camtoworld` (float32), `width` / `height` (python ints),
    `image_name`, `image_id` (the scene index).
  - The image is loaded/undistorted **lazily per access**, so a large scene
    doesn't have to fit in RAM. With `num_workers=4` in the DataLoader this
    happens on worker processes.

---

## 5. `model.py` — Gaussian parameters, optimizers, scheduler

Everything about the *representation*: how to turn a sparse point cloud into
learnable Gaussians, and how to optimize them. No training loop here.

### `SH_C0 = 0.28209479177387814`
The DC (order-0) real spherical-harmonics basis value, `1 / (2*sqrt(pi))`.
Used to convert between RGB colors and SH DC coefficients.

### `rgb_to_sh(rgb)` / `sh_to_rgb(sh)`
`rgb_to_sh: (rgb - 0.5) / SH_C0`. `sh_to_rgb: sh * SH_C0 + 0.5`. The `- 0.5`
centers color around mid-gray so the DC coefficient is signed. Exact inverses
(tested in `test_rgb_sh_roundtrip`).

### `num_sh_bases(degree) -> (degree + 1) ** 2`
Number of SH coefficients for a given max degree. Degree 0 → 1, degree 3 → 16.
Each Gaussian stores this many RGB triples.

### `_knn_distances(points, k, chunk_size=4096) -> (N,)`
Mean distance from each point to its `k` nearest neighbors, by brute-force
chunked `torch.cdist` (no scikit-learn / faiss dependency):
- For each chunk of ≤4096 points, compute `cdist(chunk, all_points)` →
  `[chunk, N]` distance matrix.
- `topk(k+1, largest=False)` — smallest `k+1` distances (the `+1` because a
  point's nearest "neighbor" is itself at distance 0).
- Drop column 0 (self), mean the rest → mean NN distance for each point.
- Chunking bounds peak memory at `chunk_size * N` floats. Comment notes this
  is fine up to a few hundred thousand points, which covers typical SfM
  clouds.

### `@dataclass GaussianModelConfig`
- `sh_degree = 3` — max SH degree (16 coeffs/color).
- `init_opacity = 0.1` — every Gaussian starts fairly transparent so
  densification/opacity optimization can build up structure.
- `init_scale = 1.0` — multiplier on the NN distance used for initial
  Gaussian size.
- Per-parameter Adam learning rates: `means_lr = 1.6e-4` (later multiplied by
  `scene_scale`), `scales_lr = 5e-3`, `quats_lr = 1e-3`,
  `opacities_lr = 5e-2`, `sh0_lr = 2.5e-3`, `shN_lr = 2.5e-3 / 20` (higher-order
  SH learns 20x slower than the DC term — standard 3DGS practice).
  These values match nerfstudio's `splatfacto` defaults.

### `init_gaussians(points_xyz, points_rgb, config, device="cuda") -> torch.nn.ParameterDict`
`points_rgb` expected in `[0, 1]`. Builds six learnable tensors, one row per
SfM point (`n` rows):

| param | shape | init | stored as |
|---|---|---|---|
| `means` | `(n, 3)` | the SfM point xyz | raw world coords |
| `scales` | `(n, 3)` | `log(knn_dist * init_scale)`, repeated to 3 axes | **log** scale (exp'd at use) |
| `quats` | `(n, 4)` | `[1, 0, 0, 0]` (identity rotation) | wxyz quaternion |
| `opacities` | `(n,)` | `logit(0.1)` | **logit** (sigmoid'd at use) |
| `sh0` | `(n, 1, 3)` | `rgb_to_sh(points_rgb)` | DC SH coefficient |
| `shN` | `(n, 15, 3)` | zeros | higher-order SH coefficients |

- `k = min(4, n - 1)` (or 1 if `n == 1`) — guards tiny clouds.
- Storing scale in log-space and opacity in logit-space means Adam operates
  in an unconstrained space while the actual values stay positive / in `[0,1]`.
- `colors` is allocated as `(n, dim_sh, 3)` zeros, DC row filled, then split
  into `sh0` (row 0) and `shN` (rows 1:). They're split so the optimizer and
  the SH-degree warmup schedule can treat them independently.
- Returns a `ParameterDict` moved to `device`.
- Covered by `test_init_gaussians_shapes` (all six shapes, identity quats,
  means untouched, opacity in `(0,1)`, scale > 0).

### `build_optimizers(params, config, scene_scale) -> dict[str, Adam]`
One `torch.optim.Adam` **per parameter** (not one optimizer with param
groups). Learning rates as in the config, except `means` is
`means_lr * scene_scale` — a bigger scene needs bigger positional steps.
`eps=1e-15` (3DGS uses a tiny epsilon so Adam's denominator doesn't wash out
already-small gradients). Separate optimizers matter because densification
adds/removes rows and each param's Adam state has to be resized independently
— which `gsplat`'s strategy does by mutating these optimizer objects directly.
- `test_optimizers_and_scheduler` asserts the six keys exist and
  `means` lr == `means_lr * scene_scale`.

### `build_means_scheduler(optimizers, max_steps) -> ExponentialLR`
Only `means` gets a schedule. `gamma = 0.01 ** (1 / max_steps)` so after
`max_steps` steps the means LR has decayed to exactly 1% of its start value.
Standard since the original 3DGS paper — positions should settle down as
training progresses. Test steps it 1000 times and checks the ratio is 0.01.

### `get_colors(params)` / `get_scales(params)` / `get_opacities(params)`
Activation helpers used at every render call:
- `get_colors`: `cat([sh0, shN], dim=1)` → `(n, 16, 3)` full SH tensor.
- `get_scales`: `exp(params["scales"])` → positive world-space scale.
- `get_opacities`: `sigmoid(params["opacities"])` → `(0, 1)`.

---

## 6. `train.py` — the training loop

The whole loop — render, loss, backward, densify, checkpoint — is one
function. Densification is **not** reimplemented; it's delegated to
`gsplat.strategy.DefaultStrategy`.

### `@dataclass TrainConfig`
- `data_dir`, `output_dir`, `colmap_path` (optional explicit sparse path),
  `downscale_factor` (`int | "auto" | None`).
- `max_steps = 30_000`.
- `eval_every = 8` — held-out split (used only to *exclude* eval images from
  training here; actual eval is a separate command).
- Loss / schedule:
  - `ssim_lambda = 0.2` — loss is `0.8 * L1 + 0.2 * (1 - SSIM)`.
  - `sh_degree_interval = 1000` — unlock one more SH degree every 1000 steps.
  - `random_background = True` — composite over a random color each step so
    the model can't cheat by baking the background into Gaussians.
- Densification knobs (all forwarded to `DefaultStrategy`, all matching
  `splatfacto` defaults): `warmup_length=500`, `refine_every=100`,
  `reset_alpha_every=30` (in units of refine steps → every 3000 steps),
  `cull_alpha_thresh=0.1`, `cull_scale_thresh=0.5`, `cull_screen_size=0.15`,
  `densify_grad_thresh=0.0008`, `densify_size_thresh=0.01`,
  `split_screen_size=0.05`, `stop_split_at=15_000`, `stop_screen_size_at=4_000`.
- `model: GaussianModelConfig` (nested, `default_factory`).
- Bookkeeping: `save_every=5_000`, `save_ply=True`, `log_every=50`,
  `device="cuda"`, `seed=42`.

### `_viewmat_from_camtoworld(camtoworld) -> viewmat`
`torch.linalg.inv(camtoworld)`. That's the whole function. The comment is the
point: COLMAP poses are already in the OpenCV convention gsplat's rasterizer
expects, so world-to-camera is a plain inverse — **no axis flip**. nerfstudio
stores OpenGL-convention poses and has to negate axes here; this pipeline
sidesteps that by never leaving COLMAP's convention. Shared with `eval.py`.

### `train(cfg) -> Path`
Step by step:

1. `torch.manual_seed(cfg.seed)`.
2. `scene = load_scene(...)`; `train_idx, eval_idx = train_eval_split(...)`.
   Print image counts, SfM point count, `scene_scale`.
3. `train_dataset = GaussianSplattingDataset(scene, train_idx)` and a
   `DataLoader(batch_size=1, shuffle=True, num_workers=4,
   collate_fn=lambda x: x[0])`. The `collate_fn` unwraps the single-element
   batch list so `batch` is the sample dict directly (batch size is always 1
   — one image per iteration).
4. Move the SfM cloud to device: `points_xyz` as float, `points_rgb` as
   `float / 255`.
5. `params = init_gaussians(...)`, `optimizers = build_optimizers(...,
   scene.scene_scale)`, `means_scheduler = build_means_scheduler(...)`.
6. **Build `DefaultStrategy`** with the config knobs. Field-name mapping:
   - `prune_opa = cull_alpha_thresh`
   - `grow_grad2d = densify_grad_thresh`
   - `grow_scale3d = densify_size_thresh`, `grow_scale2d = split_screen_size`
   - `prune_scale3d = cull_scale_thresh`, `prune_scale2d = cull_screen_size`
   - `refine_scale2d_stop_iter = stop_screen_size_at`
   - `refine_start_iter = warmup_length`, `refine_stop_iter = stop_split_at`
   - `reset_every = reset_alpha_every * refine_every` (30 * 100 = 3000)
   - `refine_every = refine_every`
   - `verbose = False`
   Then `strategy_state = strategy.initialize_state(scene_scale=...)` — the
   strategy's running buffers (per-Gaussian gradient accumulators etc.).
7. `ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3)` from
   `pytorch_msssim`.
8. `infinite_loader()` — a generator that loops `yield from train_loader`
   forever, so the training loop can pull `max_steps` samples regardless of
   dataset size. `data_iter = infinite_loader()`.
9. **Main loop** `for step in tqdm(range(max_steps))`:
   a. `batch = next(data_iter)`.
   b. Move to device and add a batch dim: `camtoworld [1,4,4]`, `K [1,3,3]`,
      `gt_image = (image / 255)[None]` → `[1, H, W, 3]`.
   c. `viewmat = _viewmat_from_camtoworld(camtoworld[0])[None]`.
   d. `sh_degree_to_use = min(step // 1000, cfg.model.sh_degree)` — SH warmup.
   e. **Render** via `gsplat.rendering.rasterization(...)`:
      - `means`, `quats` (raw), `scales = get_scales` (exp'd),
        `opacities = get_opacities` (sigmoid'd), `colors = get_colors`
        (full SH), `viewmats`, `Ks`, `width`, `height`,
        `sh_degree = sh_degree_to_use`, `packed=False`,
        `absgrad = strategy.absgrad`.
      - Returns `render [1,H,W,3]`, `alpha [1,H,W,1]`, `info` (the per-Gaussian
        2D means/radii/gradient hooks the strategy needs).
   f. `strategy.step_pre_backward(params, optimizers, state, step, info)` —
      must run *before* `.backward()` so the strategy can register grad hooks
      / retain the 2D means graph.
   g. **Background composite:** `background = rand(3)` if `random_background`
      else `zeros(3)`. `pred_image = render[..., :3] + (1 - alpha) *
      background`, then `clamp(0, 1)`.
   h. **Loss:** `l1 = F.l1_loss(pred, gt)`;
      `ssim_loss = 1 - ssim_fn(pred.permute(0,3,1,2), gt.permute(0,3,1,2))`
      (SSIM wants NCHW); `loss = 0.8 * l1 + 0.2 * ssim_loss`.
   i. `loss.backward()`.
   j. **Optimizer step:** for each of the six optimizers: `opt.step()` then
      `opt.zero_grad(set_to_none=True)`. Then `means_scheduler.step()`.
   k. `strategy.step_post_backward(params, optimizers, state, step, info)` —
      this is where Gaussians are actually split / duplicated / pruned and
      where the six optimizers' internal Adam state is resized to match.
      After this call `params["means"].shape[0]` may have changed.
   l. **Logging:** every `log_every` steps update the tqdm postfix with loss
      and current Gaussian count.
   m. **Checkpoint:** when `(step + 1) % save_every == 0` or it's the last
      step: `save_checkpoint(ckpt_dir / f"step-{step+1:09d}.pt", params,
      step+1, extra={sh_degree, scene_scale})`; if `save_ply`,
      `export_ply(output_dir / "point_cloud" / f"step-...ply", params)`;
      `tqdm.write` a line.
10. After the loop: `save_checkpoint(ckpt_dir / "final.pt", ...)` and return
    that path.

Not implemented (present in `splatfacto` behind flags): pose optimization,
bilateral-grid appearance correction, MCMC densification, scale
regularization, mask/depth supervision, LPIPS.

---

## 7. `io.py` — checkpoints and PLY export

### `save_checkpoint(path, params, step, extra=None)`
`mkdir -p` the parent, build a payload dict:
`{"step": step, "params": {k: v.detach().cpu() for ...}, **(extra or {})}`,
`torch.save` it. So `extra={"sh_degree": ..., "scene_scale": ...}` gets
flattened into the top level alongside `step` and `params`.

### `load_checkpoint(path, device="cuda") -> (ParameterDict, meta)`
`torch.load(..., map_location=device, weights_only=False)`. Rebuild
`params` as a `ParameterDict` of `Parameter(v.to(device))`. `meta` = every
top-level key except `params` (so `step`, `sh_degree`, `scene_scale`).
- `weights_only=False` because the payload has plain Python scalars/dicts,
  not just tensors. (Fine here since checkpoints are self-produced.)

### `export_ply(path, params)`
Writes a binary-little-endian `.ply` in the field layout the original INRIA
3DGS code and nerfstudio's `splatfacto` exporter use, so the scene opens in
any off-the-shelf splat viewer (e.g. the web viewers, Blender add-ons).

Per-vertex fields, in order:
- `x, y, z` — from `means`.
- `nx, ny, nz` — normals, written as `0.0` (splats have no meaningful normal;
  the field exists for format compatibility).
- `f_dc_0..2` — the DC SH coefficient (`sh0` reshaped to `(n, 3)`).
- `f_rest_0..44` — higher-order SH (`shN` reshaped to `(n, 45)`; 15 coeffs ×
  3 channels). **Note:** written in `(coeff, channel)` order as flattened by
  `.reshape` — some viewers expect `(channel, coeff)` and may need transposing.
- `opacity` — **pre-sigmoid (logit)** value, matching the 3DGS convention
  (viewers apply sigmoid themselves).
- `scale_0..2` — **pre-exp (log)** scale, same reasoning.
- `rot_0..3` — the raw quaternion (wxyz).

Builds a NumPy structured array (`dtype` list assembled to match), fills each
named column, writes an ASCII header
(`ply` / `format binary_little_endian 1.0` / `element vertex N` / one
`property float <name>` per field / `end_header`) then `elements.tobytes()`.

---

## 8. `eval.py` — held-out metrics

### `@dataclass EvalConfig`
`data_dir`, `checkpoint`, `output_dir`, `colmap_path`, `downscale_factor`,
`eval_every=8`, `device="cuda"`.

### `psnr(pred, gt) -> tensor`
`mse = mean((pred - gt) ** 2)`; `-10 * log10(mse)`. Assumes inputs in
`[0, 1]` (so max signal is 1, and the usual `20*log10(MAX) - 10*log10(mse)`
reduces to `-10*log10(mse)`).

### `evaluate(cfg) -> dict` — decorated `@torch.no_grad()`
1. `load_scene(...)`; `_, eval_idx = train_eval_split(...)`;
   `eval_dataset = GaussianSplattingDataset(scene, eval_idx)`.
2. `params, meta = load_checkpoint(cfg.checkpoint)`;
   `sh_degree = meta.get("sh_degree", 3)`.
3. `render_dir = output_dir / "renders"`, `mkdir -p`.
4. **Loop over eval views** (`tqdm`):
   - Same setup as training: move `camtoworld`, `K`, `gt_image` to device,
     add batch dim, `viewmat = _viewmat_from_camtoworld(...)`.
   - `rasterization(...)` with the **full** `sh_degree` (no warmup at eval).
   - Composite over a **black** background (`zeros(3)`), clamp `[0, 1]`.
   - `psnrs.append(psnr(...).item())`;
     `ssims.append(ssim_fn(pred_nchw, gt_nchw, data_range=1.0).item())`
     (here `ssim_fn` is `pytorch_msssim.ssim`, the functional form).
   - **Save a side-by-side:** `cat([gt, pred], dim=1)` (horizontal), ×255 →
     uint8, `imageio.imwrite(render_dir / image_name, ...)`. So each file is
     ground-truth on the left, render on the right, for eyeballing.
5. `results = {num_images, psnr: mean(psnrs), ssim: mean(ssims)}`; dump to
   `output_dir / metrics.json`; print a one-line summary; return `results`.

No LPIPS (deliberate — avoids a pretrained-weights download for a
sanity-check metric).

---

## 9. `viewer.py` — interactive viser viewer

Renders the scene live as the user orbits/pans/zooms in a browser. Built
directly on `viser`, not nerfstudio's viewer abstraction or gsplat's
`nerfview`.

### Constants
- `MAX_RENDER_WIDTH = 1600` — cap render width so a huge browser window
  doesn't tank the frame rate.
- `VISER_DEFAULT_BACKGROUND = (0.1490, 0.1647, 0.2157)` — viser's default
  dark-slate UI background, matched so composited splats blend into the
  viewport.

### `@dataclass ViewerConfig`
`checkpoint`, `device="cuda"`, `port=7007`.

### `render_from_camera(params, sh_degree, device, wxyz, position, fov, width, height) -> np.ndarray` — `@torch.no_grad()`
Pure function (no viser server dependency → directly testable):
1. **Intrinsics from FOV:** `fy = height / 2 / tan(fov / 2)`, `fx = fy`
   (square pixels), principal point at the image center. Build `K [1,3,3]`.
2. **Pose:** `R = viser.transforms.SO3(wxyz).as_matrix()`;
   `camtoworld = eye(4)` with `R` and `position` slotted in;
   `viewmat = inv(camtoworld)[None]`. Again no axis flip — viser's client
   camera is already OpenCV-convention, same as everything else here.
3. `rasterization(...)` with full `sh_degree`.
4. Composite over `VISER_DEFAULT_BACKGROUND`, `clamp(0,1)`, return
   `image.cpu().numpy()` (H, W, 3 float).

### `class _ClientRenderLoop`
One per connected browser client. A background thread that re-renders only
when that client's camera has moved.
- `__init__(client, params, sh_degree, device)`:
  - `self._dirty = threading.Event()`, initially **set** (render once
    immediately on connect).
  - `self._stop = threading.Event()`.
  - `client.camera.on_update(lambda _: self._dirty.set())` — every camera
    change flags dirty.
  - Starts `self._thread` (`daemon=True`) running `self._run`.
- `stop()` — `self._stop.set()`.
- `_run()`:
  - Loop while not stopped.
  - `self._dirty.wait(timeout=0.2)` — block until dirty or 0.2s passes;
    `continue` if it timed out (lets the loop notice `_stop`).
  - `self._dirty.clear()`.
  - Compute render size: `aspect = camera.aspect`;
    `width = min(MAX_RENDER_WIDTH, camera.image_width or 1280)`;
    `height = int(width / aspect)`.
  - `render_from_camera(...)`.
  - `client.scene.set_background_image(image, format="jpeg")` — push the
    frame as the viewport background (the whole scene is a single rendered
    image, not actual 3D geometry sent to the browser).

### `run_viewer(cfg)`
1. `load_checkpoint`; read `sh_degree` and `step` from `meta`; print Gaussian
   count.
2. `server = viser.ViserServer(port=cfg.port)`.
3. `@server.on_client_connect` → append a new `_ClientRenderLoop` for that
   client.
4. Print the URL, then `while True: time.sleep(1.0)`.
5. On `KeyboardInterrupt`: `loop.stop()` for every client loop, then return.

---

## 10. Supporting files

### `scripts/run_pipeline.sh`
`set -euo pipefail`. Reads `IMAGE_DIR` (required), `DATA_DIR`
(default `data/scene`), `OUTPUT_DIR` (default `outputs/scene`), `MAX_STEPS`
(default 30000) from the environment. Then:
1. `gsplat-pipeline sfm --image-dir $IMAGE_DIR --output-dir $DATA_DIR`.
2. Symlink `$DATA_DIR/images -> realpath($IMAGE_DIR)` if not already present
   (the dataset loader wants an `images/` folder next to `sparse/`, but `sfm`
   reads straight from `IMAGE_DIR`).
3. `gsplat-pipeline train --data-dir $DATA_DIR --output-dir $OUTPUT_DIR
   --max-steps $MAX_STEPS`.
4. `gsplat-pipeline eval --data-dir $DATA_DIR --checkpoint
   $OUTPUT_DIR/checkpoints/final.pt --output-dir $OUTPUT_DIR`.
5. Print where metrics / renders landed and the `view` command to run.

### `Dockerfile`
- Base `nvidia/cuda:12.4.1-devel-ubuntu22.04` (devel, not a prebuilt pytorch
  image — torch comes from uv per `pyproject.toml`'s CUDA index override, so
  nothing to fight).
- Installs `python3.11` + venv, symlinks `python3`.
- **COLMAP via micromamba:** Ubuntu 22.04's apt `colmap` is an incompatible
  v3.x. Creates `/opt/colmap-env` with `colmap=4.0.4` and a pinned
  `libfaiss=1.10.0=cpu_openblas*` (newer conda-forge libfaiss is ABI-broken
  against this colmap). Wraps it in a `/usr/local/bin/colmap` shell script
  that sets `LD_LIBRARY_PATH` only for that process (avoids leaking conda's
  OpenSSL into system tools like sshd).
- Copies `uv`/`uvx` from the astral image.
- `COPY pyproject.toml uv.lock README.md ./` + `COPY src ./src`, then
  `uv sync --frozen --no-dev` (installs exactly the lockfile into
  `/app/.venv`).
- `EXPOSE 7007` (viser). `ENTRYPOINT ["uv", "run", "gsplat-pipeline"]`,
  `CMD ["--help"]`.
- Comment reminder: gsplat JIT-compiles CUDA kernels on the first
  `rasterization()` call, i.e. first `train`/`view`, not at image build.

### `.github/workflows/docker-publish.yml`
`workflow_dispatch` only (manual — the image is slow to build and not needed
per-commit). Checkout → buildx → Docker Hub login (`DOCKERHUB_USERNAME` /
`DOCKERHUB_TOKEN` repo secrets) → `build-push-action` to
`rorygh/gsplat-pipeline:latest`, `linux/amd64`, with GHA layer cache.

### `pyproject.toml`
- Runtime deps: `torch>=2.1`, `gsplat>=1.4,<2`, plus `packaging` and
  `setuptools` (both imported at runtime by gsplat's JIT / torch's
  `cpp_extension` but not declared by them), `numpy`, `opencv-python-headless`,
  `imageio`, `pillow<11`, `pytorch-msssim`, `viser>=0.2`, `tyro>=0.8`, `tqdm`.
- `dev` extra: `pytest`.
- `[project.scripts] gsplat-pipeline = "gsplat_pipeline.cli:main"`.
- hatchling build, wheel packages `src/gsplat_pipeline`.
- `[tool.uv.sources] torch = { index = "pytorch-cu124" }` + an explicit
  `[[tool.uv.index]]` pointing at `https://download.pytorch.org/whl/cu124`.
  PyPI's `torch` wheel is CPU-only; gsplat compiles against whatever torch is
  active, so this must match the GPU's CUDA version (change to `cu121` etc.
  for a different driver, and match the Dockerfile base image).

---

## Cross-module contracts (quick reference)

- **Camera convention:** OpenCV (+X right, +Y down, +Z forward) everywhere —
  COLMAP native, gsplat native, viser client native. `camtoworld` is stored;
  `viewmat` is always a bare `inv(camtoworld)`.
- **`_viewmat_from_camtoworld`** lives in `train.py`, imported by `eval.py`.
  `viewer.py` inlines the same one-liner.
- **`load_scene` + `train_eval_split` + `GaussianSplattingDataset`** (all in
  `colmap/dataset.py`) are shared by `train.py` and `eval.py` unchanged.
- **Parameter storage transforms:** scale in log-space, opacity in
  logit-space, color as SH coefficients. `get_scales` / `get_opacities` /
  `get_colors` in `model.py` undo these at every render call (train, eval,
  viewer all import them). The `.ply` export writes the **stored** (pre-
  activation) values, matching 3DGS-ecosystem convention.
- **Checkpoint payload:** `{step, params, sh_degree, scene_scale}`. `eval`
  and `viewer` both read `sh_degree` from it (default 3 if missing).
- **Densification ownership:** `gsplat.strategy.DefaultStrategy` owns the
  Gaussian count and mutates the six per-parameter Adam optimizers in place;
  the training loop just calls `step_pre_backward` (before `.backward()`) and
  `step_post_backward` (after `opt.step()`).
