# Manifest policy

Status: **confirmed by the 2026-08-22 real data audit and run manifests**.

The locked machine-readable file is `clevr_3x3_manifest.json`. It is maintained by the data-adapter workflow and intentionally is not duplicated or hand-authored here by the documentation workflow.

The manifest must contain exactly:

- the three subject tokens and initializers;
- the three color tokens, initializers, and locked RGB values;
- material `metal`;
- all nine sample IDs listed in the study README;
- exactly one official-template prompt for every subject/color pair;
- enough source-path and hash provenance to detect missing, duplicate, or changed inputs.

Before a run, validation must confirm the complete 48-sample source inventory, scene labels, 512×512 RGB images, 512×512 binary masks, and the nine selected combinations. Staging may expose only `img.jpg` to the ColorPeel directory loader. Masks, scene JSON, backgrounds, and original copies must remain outside loader-visible training directories.

The baseline config references this file. A missing manifest blocks execution; it must not be generated implicitly during training.

## Conditional multiview protocol

`clevr_multiview_protocol.json` and its schema are a tracked request/validation
contract for the conditional `multiview_heldout_v1` study. They are not rendered
data and do not show that multiview training occurred. The protocol declares 20
views per cell, reserves views 0–15 for training and 16–19 for audit, and defines
three held-out matchings. Camera, lighting, background, scene, image, and mask
fields remain renderer-owned.

`prepare_clevr_multiview.py plan` must report `blocked` when no real renderer is
provided. Only `realize` may validate 180 real images/masks and derive fold
assets. Until a renderer realization and its validation record exist,
multiview rendering, fold configs, and all multiview training remain `pending`.
