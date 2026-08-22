# Comparability report

## Comparison anchor

The anchor is the official ColorPeel repository at `021f5c74cee6c231a03b8b49bb96750cadfc4e06` and the paper's coarse-color training setting. The preserved components are SD 1.4, Custom Diffusion cross-attention K/V training, learned modifier embeddings, 512 resolution, 1500 steps, LR `1e-5`, and CAA/cosine weight 0.2.

## Preserved behavior

- Backbone: `CompVis/stable-diffusion-v1-4`.
- Six total learned tokens and ordinary-word initializers.
- Official prompt form: `a photo of <subject-token> shape in <color-token> color`.
- Batch 1, seed 42, constant LR scheduler, no warmup, gradient accumulation 1.
- Optimizer and gradient clipping defaults from official code.
- Latent valid-region reconstruction mask; no GT-mask supervision.
- xFormers, prior preservation, and resume disabled.

## Meaningful deviations

- The dataset is CLEVR and the grid is 3 subjects × 3 colors, all metal.
- Token roles are 3+3 rather than the official 2+4 launcher.
- A gradient-mask defect is corrected so the last learned token is not silently frozen.
- Modifier-only text updates are an intended anchor, but the official full-parameter AdamW weight decay can modify zero-gradient non-modifier rows. The reproduction must resolve and disclose this behavior before comparability is finalized.
- Standalone sampling is made deterministic and loads all requested tokens.
- Grounded-SAM and Qwen3-VL provide evaluation adaptations.
- The main-paper batch-size statement differs from the public launcher/supplementary setting; this reproduction follows the official launcher value 1.

## Interpretation boundary

This experiment can test whether the ColorPeel mechanism transfers to the specified CLEVR grid and can expose subject/color leakage. It cannot establish exact paper-score reproduction because the data, token allocation, prompts, and parts of evaluation differ. Official color metrics and CLEVR diagnostics must be reported separately.

No result is currently available. `not_run` means no claim can be made about training stability, visual quality, color accuracy, shape accuracy, joint accuracy, or entanglement reduction.

## Required evidence for later claims

- Exact environment freeze and SD 1.4 provenance.
- Dataset staging manifest, source hashes, scene-label checks, and mask audit.
- Patch diff plus unit tests, especially all-six-token gradient behavior.
- A value-level before/after test showing that non-modifier embedding rows remain unchanged despite AdamW weight decay.
- Smoke/full training commands and complete logs.
- Checkpoint/token file inventory and reload test.
- 900-row generation manifest with prompt and seed provenance.
- Segmentation-failure ledger, official color metrics, VLM prediction rows, accuracies, and both leakage contingency tables.
