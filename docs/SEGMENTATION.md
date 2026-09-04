# Semantic & instance segmentation — method survey (not implemented)

Where we are: `mask` produces a **single binary object mask** per frame, and
`train --mask-dir` bakes it in (drop off-object points → mask the loss →
supervise alpha → prune to the object). That gives *one* segmented thing.

This note estimates the step up to **many** labels:

- **Semantic** — every Gaussian (or pixel) gets a *class* ("box", "table",
  "wall"), from a fixed or open vocabulary.
- **Instance** — every Gaussian gets an *object id* (box #1 vs box #2, same
  class).
- **Panoptic** — both at once.

Nothing here is built. It's a map of the options, their cost, and a
recommended path for this codebase.

---

## 1. Where the labels live

| Representation | How | Pros / cons |
|---|---|---|
| **2D only** (per training image), lifted on demand | run a 2D model per frame, project into 3D at query time | no change to the splat; slow queries; no 3D consistency guarantee |
| **Per-Gaussian discrete label** | one `int` (class) / `int` (instance) per Gaussian, stored in the PLY/checkpoint | tiny, renderable by majority vote, trivial to filter/export; fixed at bake time, no open-vocab |
| **Per-Gaussian feature vector** | a learned `f_i ∈ R^d` (d≈16–512) per Gaussian, alpha-composited like colour | open-vocab text query, interactive click-select, hierarchy; +10–40% train time, +memory, needs a 2D feature model |
| **Separate field** (small MLP / hash grid keyed on position) | queried alongside the splat | decouples from geometry; another network to train/serve |

For this repo the natural fit is **per-Gaussian discrete label** first
(matches how `transform`/`sh_degree` already ride in the checkpoint), with
**per-Gaussian features** as the "nice viewer" upgrade.

---

## 2. Where the 2D labels come from

You need *something* to supervise with. All of these are off-the-shelf.

**Closed vocabulary (fixed class list):**
- **Mask2Former / OneFormer / SegFormer** — semantic + panoptic, ADE20K /
  Cityscapes / COCO heads. Best when your classes match a driving/indoor set.

**Open vocabulary (text-defined classes):**
- CLIP-aligned pixel features: **LSeg, OpenSeg, CAT-Seg, SAN, FC-CLIP** —
  per-pixel embedding, cosine-sim to text prompts → class.
- **Grounded-SAM** (GroundingDINO text→box → SAM mask) — open-vocab
  *instance* masks from a prompt list.

**Class-agnostic "segment everything":**
- **SAM / SAM 2** — masks without labels, at multiple granularities. SAM 2
  adds **video propagation** → temporally consistent instance masks from one
  click, which is the key ingredient for a walkaround.
- **DEVA** — decouples image segmentation from temporal association; pairs any
  image model with cross-frame linking.

**Motion (for the dynamic-object case):**
- **SegAnyMo**, epipolar/flow residual — segment *the moving thing*; see
  `BACKGROUND_REMOVAL_PLAN.md` regime B. Its output is a per-instance moving
  mask, i.e. instance segmentation of the dynamic object for free.

---

## 3. Lifting 2D → 3D (the actual research area)

### 3a. Label-lifting by voting  *(cheapest, recommended first)*
For each Gaussian: project its centre into every training view, read the 2D
label at that pixel (gated by rendered depth/alpha so occluded hits don't
count), accumulate votes across views, take the majority. Store one label per
Gaussian.
- **No extra training**, reuses the render + pose infra we already have
  (`gaussians_in_masks` in `colmap/dataset.py` is literally this, for one
  label — generalises to N).
- **Semantic**: works directly if the 2D class maps are consistent enough.
- **Instance**: needs the 2D instance ids to already be **cross-view
  consistent** — otherwise "instance 3 in frame 1" ≠ "instance 3 in frame 2".
  Fix with SAM 2 video propagation, DEVA, or a 3D-IoU merge (lift each
  frame's masks to 3D via depth, union masks that overlap in space).
- Cost: minutes, CPU-ish. Quality: good for well-separated objects, fuzzy at
  boundaries and on thin structure.

### 3b. Per-Gaussian feature distillation  *(higher quality, open-vocab)*
Add `features` to the parameter dict; render them (alpha-composite) to a 2D
feature map; loss = distillation against a frozen 2D foundation model's
feature map (DINO / CLIP / SAM encoder). Then:
- semantic = cosine-sim of each Gaussian feature to text embeddings;
- instance = cluster features, or a contrastive loss that pulls Gaussians
  projecting into the same 2D mask together and pushes different masks apart.

Representative work:
- **Feature-3DGS** — generic feature distillation into Gaussians.
- **LangSplat** — CLIP features + a scene-specific autoencoder to shrink 512→3
  D so it's cheap to store/render; SAM for multi-scale masks.
- **LEGaussians** — quantised language features + per-Gaussian uncertainty.
- **Gaussian Grouping** — adds an "identity encoding" supervised by SAM masks
  + a 3D spatial-consistency regulariser → clean **instances**; probably the
  closest match to "segment the objects in my capture".
- **GARField** — scale-conditioned affinity field → a *hierarchy* of instances
  (part → object → group), pick the granularity at query time.
- **SAGA / SAGD / Click-Gaussian / OmniSeg3D** — interactive: click a Gaussian,
  get its whole object, via contrastive SAM-feature fields.

2025 open-vocab *instance* specifically: **OpenSplat3D**, **PanoGS**
(panoptic), **Vote-Splat** (Hough voting), **SceneSplat** (VL pretraining on
3DGS scenes), **ReferSplat** (referring expressions). These are SOTA but each
is a research codebase, not a library.

### 3c. Joint vs post-hoc
- **Post-hoc** (freeze the trained splat, learn only the label field): fast
  (minutes), safe, what LangSplat/Gaussian-Grouping mostly do.
- **Joint** (train geometry + labels together): marginally better boundaries,
  1.3–1.5× the train run, more moving parts.

---

## 4. The instance-consistency problem (the hard part)

2D instance ids are per-frame and arbitrary. Options to make them agree:

| Method | Idea | Fit here |
|---|---|---|
| **SAM 2 video** | click once, propagate the id through the ordered sequence | best for our walkarounds; frames are already ordered |
| **DEVA** | linear assignment on cross-frame mask feature similarity | good when SAM 2 loses the object |
| **3D-IoU merge** | lift each frame's masks to 3D (depth back-projection), union masks that overlap | no learned model; needs decent depth (we have it from the splat) |
| **Contrastive feature field** | learn per-Gaussian identity codes (Gaussian Grouping / GARField) | highest quality, most work |

---

## 5. Cost / complexity estimate

| Approach | New deps | Train overhead | Impl size | Output |
|---|---|---|---|---|
| 3a voting, semantic | a 2D seg model | none | ~small | per-Gaussian class in PLY |
| 3a voting, instance | + SAM 2 (video) | none | ~small–medium | per-Gaussian instance id |
| 3b distillation, semantic (LangSplat-ish) | CLIP/SAM + autoencoder | +20–40% | ~medium–large | text-queryable feature field |
| 3b distillation, instance (Gaussian-Grouping-ish) | SAM | +30–50% | ~large | click-select instances in the viewer |

---

## 6. Recommended path for this repo (when we do it)

1. **`--semantic-dir` on `train`** — accept a folder of per-image label PNGs
   (palette or 16-bit id), produced by *whatever* 2D model the user runs
   (we don't pick one). After training, vote a per-Gaussian label and write it
   as an extra PLY field + checkpoint array. Generalises `gaussians_in_masks`.
   Zero new training, ~minutes.
2. **Instance ids**: require the input PNGs to be cross-view consistent
   (document "run SAM 2 video / DEVA first"), plus an optional `--merge-iou`
   3D pass for when they aren't.
3. **Viewer**: colour-by-label toggle, click-to-isolate (project the click
   ray, read the hit Gaussian's label, hide the rest).
4. **Feature distillation** (`--feature-model dino|clip`) only if open-vocab
   text query or smooth boundaries are needed — it's the big lift.

### Connection to the moving-object goal
Instance segmentation of a *dynamic* object **is** the motion-segmentation
problem (§2). Once SegAnyMo / flow-residual gives a consistent per-instance
moving mask, feed it to the `--mask-dir` path we already built — per instance
— to get a background-free splat of each moving object. The 4D reconstruction
(the object also *deforms/moves* between frames) is a separate, larger effort
(dynamic / deformable 3DGS) and out of scope for this static pipeline.

---

## References

- SAM 2 — <https://github.com/facebookresearch/sam2> ·
  DEVA — <https://github.com/hkchengrex/Tracking-Anything-with-DEVA>
- Grounded-SAM — <https://github.com/IDEA-Research/Grounded-Segment-Anything>
- OneFormer — <https://github.com/SHI-Labs/OneFormer> ·
  FC-CLIP — <https://github.com/bytedance/fc-clip>
- Feature-3DGS — <https://feature-3dgs.github.io/>
- LangSplat — <https://langsplat.github.io/> ·
  LEGaussians — <https://buaavrcg.github.io/LEGaussians/>
- Gaussian Grouping — <https://github.com/lkeab/gaussian-grouping>
- GARField — <https://www.garfield.studio/>
- SAGA — <https://github.com/Jumpat/SegAnyGAussians>
- OpenSplat3D — <https://arxiv.org/abs/2506.07697> ·
  PanoGS — <https://arxiv.org/abs/2503.18107>
- SceneSplat (ICCV 2025) — <https://github.com/unique1i/SceneSplat> ·
  ReferSplat — <https://arxiv.org/pdf/2508.08252>
- Survey: *3D Gaussian Splatting Applications: Segmentation, Editing, and
  Generation* — <https://arxiv.org/pdf/2508.09977>
