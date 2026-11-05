# Wood spoon faceting diagnosis, 2026-11-05

Follow-up: the later [input-cause diagnostic](../wood_spoon_input_cause_v1/RESULTS.md)
exactly reproduced the historical native asset, then reproduced facets by
hardening only its alpha while preserving RGB and crop. The initial audit below
records the evidence and unresolved questions before that controlled test.

The current bowl has geometry-derived patches of face normals; the official
flat-normal renderer makes those patches visible. Pink is **not necessary or
the exclusive cause**: one original-color reconstruction using exactly the
current SAM mask also has pronounced facets. Historical original-color
BiRefNet reconstruction is visibly smoother. This does not isolate which
mask/preprocessing difference triggered the change.

## Controlled evidence

| Asset | Input | Native vertices / faces | Fitted mask IoU |
|---|---|---|---|
| Historical | original RGB + BiRefNet soft alpha | 408127 / 818898 | 0.9478146853 |
| Current pink | hue0 recolor + binary SAM | 409546 / 821522 | 0.9291082056 |
| New control | original RGB + same binary SAM | 362400 / 727258 | 0.9313578725 |

The control and pink have byte-identical 512 alpha (21965 foreground pixels),
identical 462x462 actual conditioning canvas sizes, and equal official commit,
seed42, pipeline type, model/hash receipts, encoder/hash receipt, sampler
parameters, FP16 flow dtypes and explicit-GEMM backend. The reconstruction and
review source files did not change between their code commits. This isolates
the recoloring intervention for the current SAM preparation at this seed.
Pink can still change geometry and severity; this is not evidence of zero
color effect or a universal color ranking.

All three cameras exhausted 422 evaluations without convergence. IoU measures
silhouette alignment, not bowl curvature or accepted geometry; the historical
mask differs and its IoU is not a directly comparable geometry score.

## Geometry versus rendering

The pinned official PBR renderer computes a normal from each triangle's edge
cross product and repeats that face normal at all three raster attributes.
The same normal drives illumination and the normal preview. There is no learned
normal map in this path. See the [pinned renderer](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/renderers/pbr_mesh_renderer.py).
Old/new review settings match: 512 output, SSAA1, eight peel layers, identical
forest HDRI hash, fixed SSAO seed and constant gray clay material.

A separate 1024 diagnostic shader retains each mesh and its saved camera and
uses only fixed camera-space Lambert illumination, excluding textures,
metallic/roughness, environment lighting and SSAO. New pink and new original
both retain large facets. Thus those material effects and montage resizing
cannot explain the macro-faceting. Each visible bowl triangle projects to
less than 1.4 pixels squared in this diagnostic; the large patches are not
individual huge polygons. Many small faces have similar normals within them.

Area-weighted vertex-normal interpolation did not remove the large patches.
The first diagnostic had antipodal orientation artifacts and its vertex
results are invalid for quality interpretation. A corrected diagnostic aligns
interpolated normals to the visible face hemisphere without changing geometry.
It preserves raw normals and records QC; zero normals were absent, with 2/1/0
near-orthogonal pixels for historical/pink/control. Local speckle remains, so
this simple averaging method is **not an accepted shading or geometry fix**.
It does not establish that every possible normal-processing method would fail.

## Remaining historical differences

Historical BiRefNet alpha has 256 levels; the current SAM mask has two.
Their thresholded mask IoU is 0.9807949978 (430 differing pixels). Historical
preparation resized RGBA jointly with LANCZOS; current preparation resizes RGB
with LANCZOS and mask with NEAREST. The resulting official conditioning crop
is 464x464 historically and 462x462 currently. Camera poses also differ.

These input/preprocessing differences are the next factors to isolate, not a
proven single root cause. This task did not replay historical sampling or
separately ablate alpha softness, boundary resampling and crop. A next bounded
investigation should replay the exact historical input first, then substitute
one preparation factor at a time. No automatic sweep or production change is
made on this evidence. The user's temporary pink default remains unchanged.

## Runs and validation

New results share root
`/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_faceting_audit_v1/`:

- `20261105__sam_original_control__d0b33ec__42/trellis/native_asset.pt`: the one
  new model sample, code `d0b33ec2e69d3ff3d5a5209caf17902c287f9ae9`.
- `20261105__normals__d0b33ec`: first diagnostic, invalid vertex comparison
  retained; flat-normal evidence remains usable.
- `20261105__normals_fixed__9b5493c`: corrected normals-only diagnostic, code
  `9b5493cd88b8963e7223e95c071f02cd03068e5f`.

Each code revision was locally reviewed/tested, pushed to the existing fork,
and executed from a new clean GitHub checkout with full SHA verification and
idle GPU check. Three orientation regression tests passed locally and on the
server. Asset/geometry hashes were verified unchanged; both downloaded archive
hashes passed. Independent provenance/code review and visual checks completed.

Local evidence/gallery:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/wood_spoon_faceting_audit_20261105/index.html`.
All source images remain untouched. No training, geometry smoothing,
remeshing, production-renderer patch or quality acceptance occurred.
