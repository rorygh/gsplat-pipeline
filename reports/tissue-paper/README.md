# tissue-paper — run report

COLMAP sanity report + eval metrics for the `tissue-paper` scene (handheld
~180° arc around a Kleenex box on a table), in the **+Z-up orbit frame**.

| File | What it shows |
|---|---|
| [cameras_path.png](cameras_path.png) | ~180° arc, top-down (X-Y) + elevation (X-Z). Cameras level at z ≈ 0 (±0.1), box + table ~3 below. |
| [cameras_3d.png](cameras_3d.png) | Same, in 3D. |
| [sfm_health.png](sfm_health.png) | mean track 4.5 imgs, mean reproj err 0.63 px, ~1019 obs/image. |
| [colmap_report.json](colmap_report.json) | 30/30 registered, 1 camera (OPENCV), 6,853 points, orbit tilt 122.3° from COLMAP world. |
| [eval_metrics.json](eval_metrics.json) | Held-out PSNR **26.18**, SSIM **0.852** (4 views, 7k-step checkpoint). |

## Masked (object-only) training

`train --mask-dir masks/tissue-paper/mask`, metrics on the **object region**,
held-out views:

| | gaussians | PSNR | SSIM | notes |
|---|---|---|---|---|
| unmasked, 7k steps | 267,787 | 31.52 | 0.961 | whole scene incl. table + background |
| **masked, 3k steps** | **37,177** | 30.24 | 0.957 | object only, no floaters; SfM seed 6853->1757 pts |

~ same object-region quality with **7x fewer Gaussians** and half the steps.

## Mesh

`reports/tissue-paper/mesh/` -- [tsdf_mesh_views.png](mesh/tsdf_mesh_views.png)
+ script. TSDF fusion of the masked splat's rendered depth (Open3D), **38.6 s**,
1.17 M verts, not watertight, surfaces lumpy. SuGaR proper wouldn't install
here -- see [docs/SURFACE_RECONSTRUCTION.md](../../docs/SURFACE_RECONSTRUCTION.md).

## Read of this run

- Clean SfM despite the soft/motion-blurred footage — heavy print + wood-grain
  texture and matte surfaces make it an easy subject.
- A proper orbit (unlike car-walkaround's partial arc with a moving
  background), so the box reconstructs sharply even at 7k steps. The
  protruding tissue is blobby (thin/translucent) and the background has
  floaters (under-constrained from a single ~180° pass) — both expected.
- Good demo of the +Z-up alignment: the elevation panel shows the camera ring
  flat at z = 0 with the object below, exactly matching the "orbit in the xy
  plane" assumption.
