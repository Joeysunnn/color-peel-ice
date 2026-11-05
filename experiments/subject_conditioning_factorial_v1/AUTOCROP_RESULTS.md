# Per-alpha automatic crop rerun

2026-11-05. All16 fresh reconstructions and64 shared-camera views completed,
eight reconstructions per object, without generation failures or retries.
This rerun changes only input cropping. It does not train branches or alter
production defaults, and quality remains pending user review.

Producer: `66ab6fdb0b33992f6e3e436d9131e42928abea88`, branch
`codex/subject-conditioning-factorial-v1`, clean checkout cloned from the
authorized GitHub fork.9 relevant tests passed locally and on the server;
independent code review also checked the real16 source RGBA inputs.
GPUs0-3 were idle before execution and had no compute processes afterward.

## Input contract

Each job calls the pinned official `pipe.preprocess_image` with no
`--fixed-crop` override. The crop is computed from that job's geometry alpha
using `A > 204`; full soft values are retained for RGB*alpha compositing.
Fractional crop coordinates, official Pillow handling and out-of-canvas
padding are preserved. No new SAM/BiRef inference or RGB resizing occurred.

| Object | Geometry alpha | Automatic XYXY | Actual condition size |
|---|---|---|---|
| Mailbox | SAM binary | `[145,254,439,548]` |294x294|
| Mailbox | BiRef soft | `[144,253,440,549]` |296x296|
| Wood spoon | SAM binary | `[24,20.5,486,482.5]` |462x462|
| Wood spoon | BiRef soft | `[23,19,487,483]` |464x464|

All16 serialized RGBA, alpha and pre-alpha RGB images are unchanged from the
first fixed-crop factorial run. Seed42, model revisions, sampling, precision
and encoder remain unchanged. Original/SAM-scope and original/BiRef-scope
remain duplicate conditions; each object has8 fresh calls and6 distinct
conditioning images/native assets. Scope continues to control pink recoloring
only: outside it, original RGB remains. Thus pink/SAM-scope/soft-alpha still
differs from the older diagnostic with black RGB outside SAM.

## Verification

All16 actual model inputs match their separately prepared expected pixels,
including mode and dimensions. Generation records contain no diagnostic
fixed-crop field and agree on the full producer Git and inference config.

13 CPU native comparisons passed, with exact inputs/configurations, all five
native arrays and voxel metadata, without alignment or numerical tolerance:

- Four original-scope duplicate pairs, two per object.
- Eight soft-alpha results against the preceding fixed-crop factorial run.
- Mailbox `pink__sam__sam` against the historical hard-alpha replay.

The last result restores the user's previous hard-alpha baseline exactly:
1134277 vertices,2249226 faces; native SHA256
`8340d957fb4ea117b991c2f95c489f250000352ff542b1e0f7eb3e7383213735`.

The initial expected-PNG file-hash check against that historical condition
failed because the encodings differ; decoded mode, dimensions and every pixel
match. This is recorded in `input_factor_audit.json`. No model retry or
scientific-input adjustment followed that check.

Main comparisons use the same frozen reference and three canonical cameras
as the preceding run, with opaque native geometry and fixed Lambert lighting.
No mesh smoothing, alignment or geometry edits are applied. Separate fitted
cameras remain labeled:2 fits converged and14 reached the existing bounded
budget. Neither fit convergence nor these reproduction checks certify quality.

## Artifacts

Server run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/subject_conditioning_factorial_v1/20261105__mailbox_wood_autocrop__66ab6fd__42`.

Native assets:
`{object}/variants/{color}__{scope}__{alpha}/trellis/native_asset.pt`.

- `status.json`:16 generation jobs and2 completed common-camera render stages.
- `input_factor_audit.json`: unchanged RGB/alpha inputs and automatic crops.
- `numerical_results.json`: sizes, native hashes, counts and fitted metrics.
- `{object}/duplicate_original_*.json`:4 duplicate comparisons.
- `{object}/previous_soft_*.json`:8 comparisons with the previous run.
- `mailbox/historical_pink_sam_replay.json`: exact hard-alpha baseline replay.
- `{object}/inputs/`, `variants/`, `geometry_views/`: receipts and images.

Local gallery:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/subject_conditioning_autocrop_20261105/index.html`.

The previous gallery and results remain unchanged. New figures label each
cell's actual input dimensions and automatic crop coordinates; shared display
crops are independent of input preprocessing. Native files stay on the server.
Downloaded evidence archive:656 files,20005641 bytes, SHA256
`400d5684582f6973126b46f7ad3dfb180c3d662a608d81a397ef8a12a50845a2`.
