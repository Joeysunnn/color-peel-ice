# Two-object renderer smoke and realization

Date: 2026-09-01 (Asia/Shanghai)

## Entry-point correction

The first Blender attempt at commit `f285052` stopped before creating rendered
artifacts because the renderer imported the preparation module, which requires
Pillow unavailable in Blender's bundled Python. Commit `1cce8e7` moved only the
pure request validator into the dependency-free renderer contract. Profile,
requests, RNG, assets, rendering and training semantics did not change.

- Profile SHA-256: `75af0818d2e18a033878f050a1cf1c4791dc519c1c187eddbe7fa6a639d3f14a`
- Request SHA-256: `e9deb85f28a38cbceddf3bfc55db7527533a6c79abceef7dcb96f1069031211a`
- Patch-related server tests: `24 passed in 0.57s`
- Earlier complete server suite: `149 passed in 26.62s`

## Real Blender smoke

Run root:

`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_two_object_subject_color_material/20260901-111100__clevr_two_object_subject_color_material__render-v4-smoke__1cce8e7__520000`

- Two requests: `pair_00_forward:view_00` and `pair_00_swapped:view_00`.
- Both use render seed `520000`, identical camera and identical lights.
- Blender 4.2.11, Cycles CUDA, one Tesla V100-SXM2-32GB.
- RGB is 512x512; both instance masks are strict binary, disjoint and internal
  to the frame; the background mask is their exact complement.
- Left/right projected positions agree with the semantic side labels.
- `smoke_validation.json` SHA-256:
  `06c59724919774fa724cab29130aeb3e8d481c75f16cf98e6ae35bf4efb8c857`.
- Contact sheet SHA-256:
  `e7c5f78109324019ff5287eda2fa2dca68fc25456f17bec3734494dfd6e05e2b`.

## Full render and realization

Run root:

`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_two_object_subject_color_material/20260901-111500__clevr_two_object_subject_color_material__render-v4-full__1cce8e7__520000`

- Renderer: `360/360` succeeded, zero failed status rows.
- Render contract SHA-256:
  `25ad8a43e566d721d7a3c004eb095d68d2b41509889184fe4c33edfd128d683e1`.
- Realization: `validated_pending_human_review`.
- 18 semantic states; each has 32 image/mask pairs.
- Training staging: 576 image-only JPEG links plus 576 separate PNG masks.
- Human review: 360 rows, one 18-scene contact sheet and nine orientation
  pair sheets.
- Main contact sheet SHA-256:
  `561d1dedb51d127da754d5997fc89f864eaa50baadcb379ac53a1340d4498804`.
- Concepts manifest SHA-256:
  `34f188e9287d4eb3e3e7774cb4722ac85a8af6e56b071be8e2a2ffe306137f29`.

Automated validation and preliminary visual inspection found no clipping,
merged instances or side reversal. Some views have ordinary perspective
occlusion while both instances remain identifiable. The tracked gate remains
`training_authorized: false`; user review is required before any smoke or
1500-step training.
