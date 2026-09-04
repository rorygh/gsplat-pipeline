# aukerman — aerial drone run report

The pipeline run **unchanged** on a nadir aerial drone dataset, to check it
isn't secretly assuming an object-centric orbit.

| | |
|---|---|
| Source | [OpenDroneMap `odm_data_aukerman`](https://github.com/OpenDroneMap/odm_data_aukerman) — 77 photos, DSC, 4896×3672, a rural park (car park + building + pond + tree line) |
| Prep | resized to 1600 px long edge (`cv2.INTER_AREA`, q95) into `data/aukerman/images/`; nothing else |
| SfM | `sfm --matching-method exhaustive`, GPU — **76/77 registered**, 1 OPENCV camera, 33,313 points, mean track 7.1, 0.60 px reproj err |
| Train | `train --max-steps 7000` (default settings), ~13 min, ~250k gaussians |
| Eval | PSNR **22.34**, SSIM **0.607** (10 held-out views) |

| File | Notes |
|---|---|
| [cameras_path.png](cameras_path.png) | The lawnmower flight pattern (top-down X-Y); elevation (X-Z) shows all cameras in one plane at z≈0 looking straight down, terrain ~4 units below. |
| [cameras_3d.png](cameras_3d.png) | Same in 3D. |
| [sfm_health.png](sfm_health.png) | Much better connectivity than the orbit datasets (mean track 7.1 vs 3.8/4.5). |
| [eval_metrics.json](eval_metrics.json) | |

## Read of this run

- **The `--align` orbit frame generalises**: the circle fit degenerates on a
  grid pattern and falls back to the centroid, but the plane fit + up-sign
  (from camera-down vectors) still recovers **+Z up** correctly — cameras end
  up level at z≈0, terrain below. `orbit_tilt_deg` 171° just means COLMAP's
  arbitrary frame happened to be near-upside-down.
- **Quality is modest (PSNR 22)** and that's the data, not the pipeline: nadir
  drone imagery has a thin baseline between overlapping frames, and most of
  the frame is near-repetitive grass / tree canopy that 3DGS renders as a
  blurry, floater-y mess (dominates the metric). The built structures — car
  park, building, paths, parked cars, pond — reconstruct crisply (see
  `outputs/aukerman/renders/`).
- No masking here (there's no single object to segment).
