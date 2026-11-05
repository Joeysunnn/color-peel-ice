# Mailbox deformation isolated to the RGB preparation change

2026-11-05. For this Mailbox at seed42, replacing only binary alpha with the
saved BiRefNet alpha changes the mesh, but **does not reproduce the earlier
severe backplate/tube displacement and crossed structure**. With soft alpha
fixed, changing to the new RGB preparation reproduces that failure under both
tested crops. This is an agent visual diagnosis, independently reviewed, not
user quality acceptance or a general rule about soft alpha.

## Exact baseline and alpha-only intervention

The baseline is the previous five-color SAM-pink run `6726dab`, not the older
manually repaired historical pink asset. Its original photograph, semantic
SAM, pink RGB, full512 canvas, crop `[145,254,439,548]` and seed42 are fixed.
The override preserves the official float32 RGB*alpha operation, uint8
truncation and out-of-canvas black padding. Production default preprocessing
is unchanged.

Hard replay exactly matches the prior actual294x294 conditioning image,
model/configuration receipts, all five native arrays and voxel metadata.
Even `native_asset.pt` SHA256 is identical:
`8340d957fb4ea117b991c2f95c489f250000352ff542b1e0f7eb3e7383213735`.

The alpha-only input copies every old RGB byte and replaces only alpha with
the saved soft component. Old SAM has42397 pixels; within it soft alpha is255
at1677 pixels,254 at40576, and143-253 at144. In the fixed crop,40718 RGB pixels
change after compositing:40577 have maximum channel change1 and141 have larger
changes (maximum75). The2434 added alpha-support pixels have frozen black RGB,
so cannot change conditioning under the fixed crop; CPU equality verifies this.
This test changes both interior and boundary opacity, not either separately.

## Bounded follow-up: RGB x crop with identical soft alpha

Because alpha alone did not reproduce the target symptom, two additional
runs completed this comparison. All cells use identical512 soft alpha,
original photograph, SAM, seed42, pinned models, encoder, sampling, precision
and backend. Configuration equality and all actual conditioning pixels were
verified, including the existing prior run. The new RGB/new crop cell is
historical `87efe3a`, not a newly generated `d56d015` result.

| Soft-alpha RGB / crop | Native vertices / faces | Observed relative structure | Fitted mask IoU | Camera evaluations / success |
|---|---:|---|---:|---|
| Old RGB / old294 (`soft_only`) |927662 /1862758|No earlier severe crossed-structure failure|.7856212394|226 /true|
| Old RGB / new296 (`soft_new_crop`) |907542 /1816634|No earlier severe crossed-structure failure|.7807158510|330 /true|
| New RGB / old294 (`soft_new_rgb`) |1087296 /2166648|Severe displacement/crossing and extra geometry|.8319051415|422 /false|
| New RGB / new296 (prior `87efe3a`) |1033002 /2067646|Severe displacement/crossing and extra geometry|.8288188416|422 /false|

Old crop: `[145,254,439,548]`; new crop: `[144,253,440,549]`.
Hard replay has1134277 vertices /2249226 faces,270 camera evaluations with
success, and SAM-target IoU .7804528027. Its target differs from the soft
rows, so this is not a strict common-target quality ranking with them.

`geometry_factorial/` renders all five assets with identical saved old camera
matrices plus three shared centered cameras. It uses opaque native triangles,
face normals and fixed Lambert shading: no learned opacity, PBR, SSAO,
individual recentering, alignment, scaling, smoothing or refitting. Native
geometry hashes are checked before/after. The relative-structure failure
remains visible in common views, so it is not merely the fitted camera.

## What changed in RGB, and what remains unresolved

`experiments/subject_pink_input_repro_v1/prepare.py` uses
`derive_repaired_alpha(alpha > 0)` and the same support to decide editable
RGB pixels. With soft conditioning alpha, that support comes from BiRefNet
instead of the old binary SAM. The hue remains0 and the recoloring algorithm
remains unchanged, but the spatial support and edge blending change.

New versus old RGB differs at3343 full512 pixels:2434 in newly included
support,909 inside old SAM. With the same soft alpha and old crop, actual
conditioning differs at3124 pixels, maximum channel difference252 on0-255.
Thus this is not merely a binary-to-soft alpha replacement or a changed hue.
The native Mailbox is already512; RGBA resizing is not the explanation here.

The experiment localizes a sufficient trigger to this **combined RGB
construction change**, conditional on soft alpha and either tested crop.
It does not distinguish added-support RGB from altered boundary blending,
prove a universal TRELLIS mechanism, or establish either retained mesh as
fully correct. Higher silhouette IoU did not imply better relative3D
structure in these controls; the current selection score is not sufficient
evidence of geometry quality.

The supported repair direction is to separate RGB preparation from geometric
alpha changes and preserve both controls in provenance. No production
recolor/matting/selection change is made by this diagnostic. No further
factor split, seeds, model downloads, S11 expansion or training was run.

## Provenance and available outputs

Four new native meshes, no failures/retries, using clean GitHub checkout:
`d56d01528a10987d4b267b46e5ef7edd3fc7c5a5`, branch
`codex/mailbox-alpha-control-v1`.28 relevant tests passed locally and on the
server. All runs used idle-checked GPU0; no unrelated processes were stopped.

Server run root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/mailbox_alpha_control_v1/20261105__fixed_rgb_crop__d56d015__42`.

Each of `hard_replay`, `soft_only`, `soft_new_crop`, `soft_new_rgb` contains
`trellis/native_asset.pt`, `actual_condition.png`, `source_cutout512.png`,
`review/` and `repro_outputs/`. Native SHA256:

- Hard replay: `8340d957fb4ea117b991c2f95c489f250000352ff542b1e0f7eb3e7383213735`.
- Soft only: `6e5a042df2ef935641b8f1f36a11af55ef527bec97dbd6b43c2c6c6052f88634`.
- Soft/new crop: `138808d6c12e5a40316813a9a20f3c716e1e8683e8ef1610f524c893cd6c4a56`.
- Soft/new RGB: `24603e0baa1aed92614dbf2fd935204d6207ddc61c2420e5337b8b100eb4046fb`.

Audit files: `inputs/input_audit.json`, `hard_replay_comparison.json`,
`soft_only_comparison.json`, `factorial_audit.json`,
`geometry_factorial/status.json`. Initial `status.json` describes the first
two-run stage; `factorial_audit.json` records the complete four-run scope.

Local gallery:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/mailbox_alpha_control_20261105/index.html`.
The common crop used solely for figure layout is recorded in
`figure_manifest.json`; full-canvas renders are retained.

Downloaded archive SHA256:

- Initial controls: `0d55f24f712e6199a1bad877ab44f112f54a6e4c7f5a300cdc99a3985cf5d54a`.
- Follow-up controls: `f7f46a2576a3507214a498cf9d785d5668ffeeb1497506241aa446956731ea4f`.
