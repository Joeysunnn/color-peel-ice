# Isolate the wood spoon input change that triggers faceting

User-authorized diagnosis, 2026-11-05. Production defaults stay unchanged.
Previous evidence: historical original RGB + BiRefNet is smooth, but both
pink and original RGB with the current SAM mask have large bowl facets. Model,
sampling and renderer receipts match; color alone is not the explanation.

This is a sequential diagnostic, not an automatic grid search:

1. Replay the exact historical source receipt with current generation code,
   pinned models and seed42. Compare decoded inputs, config and native tensors
   exactly using `experiments.subject_pink_input_repro_v1.compare`. If replay
   differs, investigate runtime/source provenance before blaming input masks.
2. If replay succeeds, change only historical 512 alpha to `(alpha > 204)*255`,
   retaining historical RGB on its new support. This threshold is the official
   TRELLIS crop threshold, so its thresholded bbox/crop stays exactly unchanged.
   Zero newly transparent RGB to satisfy the existing receipt invariant.
   The new mask is diagnostic only, not an approved replacement segmentation.
3. Select the next single-factor comparison from the measured result. Keep the
   original photo, seed, model weights, samplers and generation code fixed;
   record any crop, RGB or alpha difference explicitly. Stop broad exploration
   once a reversible input intervention explains the observed faceting.

Historical source receipt:
`/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons_v1/20261028-013740__assets__2943bf6__42/inputs/wood_spoon/source_receipt.json`.

Original photograph SHA256:
`b17fb667be977d2ed879f3c216011e6dde8324951a6f331136a7c52ab2bc4f9d`.

Historical cutout SHA256:
`c54251c91afc98ba3e9f626a71ac234f1350dc239bc36aedfd846295da168842`.

Use fresh external outputs below
`/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_input_cause_v1/`.
Code must be reviewed/tested locally, committed/pushed, then run from a new
clean GitHub checkout with full SHA and GPU ownership checks. No source-copy
deployment, model download, FP32 comparison, asset deletion, training or
automatic quality acceptance is included.
