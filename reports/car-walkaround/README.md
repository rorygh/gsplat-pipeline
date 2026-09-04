# car-walkaround — run report

COLMAP sanity report + eval metrics for the `car-walkaround` scene, in the
**+Z-up orbit frame** (`--align`, the default). `outputs/` and `masks/` are
gitignored, so this is copied in for reference.

| File | What it shows |
|---|---|
| [cameras_path.png](cameras_path.png) | Camera centres in capture order — top-down (X-Y) and elevation (X-Z), red = view direction, grey = SfM points as context. |
| [cameras_3d.png](cameras_3d.png) | Same, in 3D. |
| [sfm_health.png](sfm_health.png) | Points-per-image, track-length histogram (mean 3.8), reprojection-error histogram (mean 0.48 px). |
| [colmap_report.json](colmap_report.json) | 50/50 images registered, 1 camera (OPENCV), 16,239 points, orbit tilt 107.8° from COLMAP world, per-image stats. |
| [eval_metrics.json](eval_metrics.json) | Held-out PSNR 18.13, SSIM 0.577 (7 eval views) — from the **pre-alignment** 30k checkpoint; re-train to refresh. |

## Read of this run

- SfM is healthy: everything registered, sub-pixel reprojection error, one
  consistent intrinsic.
- The alignment recovered a clean orbit frame — the elevation panel shows the
  cameras sitting in a level plane (z ≈ 0, ±0.12) looking ~20° down at the
  object, which sits ~1.8 below.
- But the **capture is a ~270° arc, not a full orbit** (the grey chord in the
  top-down panel spans the unobserved wedge), and **the background changed**
  between the two passes down the street — hence the modest PSNR and the
  smeared render. Fixes, in leverage order: full 360° orbit at 2–3 heights;
  mask to the car (`gsplat-pipeline mask --prompt "a car"`); handle the
  reflective glass (`docs/SURFACE_RECONSTRUCTION.md` for the mesh route).
