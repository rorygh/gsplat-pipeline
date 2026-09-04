# Surface reconstruction — SuGaR and where it fits

This pipeline produces a **radiance field** (a cloud of 3D Gaussians), not a
**surface** (a triangle mesh). A splat renders well from novel views but has
no explicit geometry: no watertight mesh, no clean normals, nothing to import
into Blender / a game engine / a CAD tool, nothing to 3D-print. For an
object-centric capture (which is exactly what `mask` + the orbit-frame
alignment target) a mesh is often the actual deliverable.

[SuGaR](https://github.com/Anttwo/SuGaR) (Guédon & Lepetit, CVPR 2024) is the
best-known way to get a mesh *out of* a Gaussian Splatting reconstruction.

## What SuGaR does

1. **Vanilla 3DGS**, ~7k iterations — an ordinary splat optimisation, same as
   `train` here but shorter.
2. **Surface-alignment regularisation**, ~2k+ iterations — adds a loss term
   pushing Gaussians to be flat (one scale ≪ the other two) and to sit *on*
   the real surface, with density that looks locally like a thin shell rather
   than a fuzzy volume. This is the paper's core contribution.
3. **Mesh extraction** — sample points on the level set implied by the
   aligned Gaussians (cheap, because they now behave like a surface), then
   **Poisson reconstruction** → a triangle mesh. Fast (minutes) and scalable,
   vs. the hours a Marching-Cubes-on-a-dense-grid approach costs.
4. **Joint refinement** — bind new Gaussians to the mesh triangles and
   optimise mesh + Gaussians together (2k–15k iters). Output is a **hybrid**:
   a mesh for geometry + per-triangle Gaussians for view-dependent
   appearance.
5. **Textured-mesh export** (optional) — a plain `.obj` + texture for tools
   that don't want the Gaussians.

End to end ≈ 30–60 min on one GPU, i.e. comparable to a full `train` run here.

Follow-up: **[Gaussian Frosting](https://anttwo.github.io/frosting/)** (Guédon
& Lepetit, ECCV 2024) reuses SuGaR's mesh extraction but wraps the mesh in a
variable-thickness "frosting" layer of Gaussians — better for fuzzy/furry/
semi-transparent material (hair, grass, foliage) that a hard surface can't
represent. Same author, same license.

## The catch: license

SuGaR is built as "a wrapper of a vanilla 3D Gaussian Splatting model" and
**vendors the original INRIA 3DGS code**, including
`diff-gaussian-rasterization`. So it inherits the
[INRIA / MPII Gaussian-Splatting research license](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md):

> The *Software* may be used "non-commercially", i.e., for research and/or
> evaluation purposes only.
> THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE *SOFTWARE* FOR COMMERCIAL
> PURPOSES WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS.

This is precisely the coupling this repo was built to avoid — we're on
`gsplat` (Apache-2.0) with a hand-written COLMAP reader specifically so there
is no INRIA-licensed code in the tree. **Dropping SuGaR in as-is would
relicense the mesh path non-commercial.** Options, cleanest first:

| Option | License | Effort |
|---|---|---|
| Run stock SuGaR as a **separate, optional tool** on the exported `.ply` — never import it into this package | its own (non-commercial) | low; a documented external step |
| Reimplement just the **regulariser + Poisson extraction** on top of `gsplat` (the method is not large; the license is on the *code*, not the ideas) | Apache-2.0 | medium |
| Use a **gsplat-native mesh path** instead — see below | Apache-2.0 | low–medium |

## Apache-2.0 alternatives worth considering first

- **2D Gaussian Splatting (2DGS)** — Gaussians are flat discs by
  construction, so they align to surfaces without SuGaR's extra stage.
  `gsplat` ships `gsplat.rendering.rasterization_2dgs` (v1.5) with depth +
  normal + distortion outputs, so this is the natural fit: swap the
  rasterizer call in `train.py`, add the depth-distortion + normal-
  consistency regularisers from the 2DGS paper, then TSDF-fuse the rendered
  depth maps into a mesh with Open3D
  (`open3d.pipelines.integration.ScalableTSDFVolume`) or marching cubes. This
  is what nerfstudio's `splatfacto` mesh export and most Apache-licensed
  "gaussians → mesh" repos do.
- **GOF (Gaussian Opacity Fields)** / **RaDe-GS** — higher-quality depth →
  mesh, both MIT/Apache-ish; heavier to integrate.
- **TSDF fusion of the current 3DGS depths** — the crudest option: render
  depth from every training view with the existing 3D (not 2D) splat,
  TSDF-fuse. Works today with zero new training code, but 3DGS depth is noisy
  where Gaussians are fat (reflections, low texture), so the mesh is rough.

## Experiment (2026-09, `tissue-paper`)

### SuGaR proper — **install blocked on this box**
`Anttwo/SuGaR` needs a Conda env pinning **pytorch3d 0.7.4 + PyTorch 2.0.1
(cuda 11.8 conda build) + python 3.9**. On a CUDA 12.4 / torch 2.4 host with
only `micromamba`:
- the `pytorch3d` + `cuda-toolkit=11.8` + `open3d` + `jupyter` solve is
  over-constrained (unsolvable);
- the minimal `python=3.9 pytorch=2.0.1=*cuda11.8* pytorch3d=0.7.4` solve
  succeeds, but the resulting `torch` fails to import —
  `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` (the conda PyTorch
  build wants an old MKL/ittnotify that nothing else will co-install), and
  every MKL-pin attempt breaks a different package.

This is exactly the ABI fragility [ARCHITECTURE.md](../ARCHITECTURE.md) cites
for avoiding `pycolmap` / the INRIA rasterizer. **Conclusion: don't vendor
it.** Run it in its own known-good Docker image as an external step if needed.

### "Something like it" — TSDF fusion of the splat's own depth  (Apache, works)
`scratchpad/mesh_tsdf.py`: render RGB + expected depth
(`rasterization(..., render_mode="RGB+ED")`) from every training view,
Open3D `ScalableTSDFVolume.integrate`, keep the largest component.

| | mask-trained `tissue-paper` splat (37k gaussians, 26 views) |
|---|---|
| Time | **38.6 s** (0.9 s render on GPU, 37.7 s TSDF+extract on CPU) |
| Output | 1.17 M verts / 2.18 M tris at 6 mm voxel (over-tessellated — raise the voxel) |
| Watertight | no |
| Quality | box shape + proportions + the tissue-tuft opening are correct; **surfaces are lumpy and holey** — 3DGS depth is volumetric, not surface-aligned, so the fused shell is noisy. This is the gap SuGaR's alignment regulariser closes. |

Masking first matters a lot: on the **unmasked** splat the TSDF volume also
tries to fuse the table + background terrain, blowing up to ~40 GB RAM. The
`--mask-dir` object-only splat meshes in a bounded volume, fast.

## `gsplat-pipeline sugar` -- SuGaR-lite, built here  (Apache, `src/sugar.py`)

A compact re-implementation of SuGaR's *ideas* on top of `gsplat`, added as a
**separate subcommand** that never touches `train.py`:

- **`sugar align`** -- loads a trained checkpoint and runs a short refinement
  (default 3000 steps) with the photometric loss plus:
  - *flatten*: drive each Gaussian's thinnest axis toward zero, with a
    `relu(max_axis - init)` term + a hard scale clamp so the discs don't blow
    up to compensate (an early version without these produced 5-unit sheets
    that made TSDF fusion OOM);
  - *opacify*: `min(op, 1-op)` -> opaque shells;
  - *normal consistency*: render each Gaussian's thin-axis direction and align
    it with the normal implied by the rendered depth (2DGS/GOF trick, no SDF
    sampling).
  With `--mask-dir` it also prunes to the object at the end. -> `surface.pt`.
- **`sugar mesh`** -- `--method tsdf` (render depth, Open3D `ScalableTSDFVolume`)
  or `--method poisson` (sample the flat discs, normals = the thin axis
  oriented outward, screened Poisson).
- **`sugar full`** -- align then mesh.

### Experiment (`tissue-paper`, masked 3k-step checkpoint, A4000)

| | |
|---|---|
| `sugar align`, 1500 steps | **~2 min**; flatten metric 0.16 -> ~0, normal-consistency 0.28 -> 0.19, L1 held at 0.003 (photometry preserved) |
| `sugar mesh --method tsdf` (voxel 5 mm) | ~1 min, 1.69 M verts, not watertight |
| `sugar mesh --method poisson` | **hung** -- Open3D 0.19's `create_from_point_cloud_poisson` never returned on this box for ~100 k points (tried 3×). Works elsewhere; needs revisiting or swapping for `pymeshlab` / a CUDA Poisson. |

Result render: `reports/tissue-paper/mesh/sugar_aligned_tsdf_views.png`.
Honest read: the alignment **converges and does flatten the Gaussians**, but
the mesh is only marginally cleaner than plain TSDF here -- because (a) the
base splat is an undercooked 3k-step run, (b) TSDF fuses *rendered depth*,
which stays noisy regardless of Gaussian shape, and (c) the payoff path
(Poisson straight off the aligned discs, using their orientations as normals
-- SuGaR's actual extraction step) is the one that hangs. A fair test wants a
full 30k splat + a working Poisson backend.

## Recommendation

1. **Best mesh today:** `sugar full --method tsdf --mask-dir ...` on a
   **full-length (30k)** checkpoint. Fix Poisson (swap Open3D for pymeshlab or
   a GPU implementation) to unlock SuGaR's real extraction quality.
2. **Bigger win:** `train --mesh` that swaps in `rasterization_2dgs` for the
   last few thousand steps -- 2D Gaussians are surface-aligned *by
   construction*, so the flatten regulariser becomes unnecessary and the
   depth itself is far less noisy.
3. Upstream SuGaR from its own Docker image if the non-commercial licence is
   acceptable and you need its published quality.

## References

- SuGaR — <https://github.com/Anttwo/SuGaR> ·
  [arXiv:2311.12775](https://arxiv.org/abs/2311.12775) ·
  [project page](https://imagine.enpc.fr/~guedona/sugar/)
- Gaussian Frosting — <https://anttwo.github.io/frosting/> ·
  [arXiv:2403.14554](https://arxiv.org/pdf/2403.14554)
- INRIA 3DGS license — <https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md>
- 2DGS — <https://surfsplatting.github.io/> ; `gsplat` 2DGS API —
  <https://docs.gsplat.studio/main/apis/rasterization.html>
- GOF — <https://niujinshuchong.github.io/gaussian-opacity-fields/> ;
  RaDe-GS — <https://baowenz.github.io/radegs/>
