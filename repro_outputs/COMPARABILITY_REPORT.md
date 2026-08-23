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
- The official full-parameter AdamW weight decay may modify zero-gradient ordinary vocabulary rows. This reproduction preserves that literal behavior, records its drift, performs no restoration, and does not use the drift as a failure criterion.
- Standalone sampling is made deterministic and loads all requested tokens.
- Grounded-SAM and Qwen3-VL provide evaluation adaptations.
- The main-paper batch-size statement differs from the public launcher/supplementary setting; this reproduction follows the official launcher value 1.

## Interpretation boundary

This experiment can test whether the ColorPeel mechanism transfers to the specified CLEVR grid and can expose subject/color leakage. It cannot establish exact paper-score reproduction because the data, token allocation, prompts, and parts of evaluation differ. Official color metrics and CLEVR diagnostics must be reported separately.

Training stability and update coverage are now supported for this adaptation:
the two real smokes and full 1500-step run completed with finite loss records,
the six modifier tokens received their expected gradients, final weights
reloaded, and literal official AdamW drift was observed. Generated-image
quality, color metrics, shape/color/joint accuracy, and entanglement reduction
remain outside this training-only evidence until the independent downstream
stage manifests close.

Those manifests have now closed. The 900 images are valid; Grounded-SAM
accepted 588/600 transfer masks and explicitly rejected 12; Qwen returned
300/300 valid predictions. Grid shape/color/joint accuracy was
94.44%/93.89%/93.89%. The single-axis contingency tables show strong biases,
so the experiment supplies mixed disentanglement evidence rather than a
binary success claim. These CLEVR/Qwen diagnostics are adaptations and are not
directly comparable to ColorPeel paper scores.

## Required evidence for later claims

- Exact environment freeze and SD 1.4 provenance.
- Dataset staging manifest, source hashes, scene-label checks, and mask audit.
- Patch diff plus unit tests, especially all-six-token gradient behavior.
- Observation outputs for the two-step smoke: exact first-two exposures, learning signals only for seen tokens, modifier deltas, and descriptive ordinary-vocabulary drift.
- Observation outputs for the independent nine-step smoke: exposure count 3, at least one nonzero-gradient step, and nonzero delta for every modifier token, plus descriptive ordinary-vocabulary drift.
- Full training command and complete log after both smoke records are reviewed.
- Checkpoint/token file inventory and reload test.
- 900-row generation manifest with prompt and seed provenance.
- Independent Grounded-SAM command/environment/model provenance, mask manifest, and segmentation-failure ledger.
- Independent Qwen3-VL command/environment/model provenance, prediction rows, and failure ledger.
- Official color metrics, CLEVR accuracies, and both leakage contingency tables after their prerequisite external stages complete.

## Diagnosis-first comparison rules

The report-01 checkpoint and generation are frozen. New diagnostics must cite
their source image IDs, prompts, seeds, checkpoint directory, training commit,
and their own distinct output directory. A checker-disabled or FP32 rerun is a
diagnostic condition, not a replacement baseline sample.

Human review is the primary semantic comparator for the follow-up. The current
review is qualitative and therefore cannot provide a numerical win rate. A
future blinded review must retain its 540-image protocol, completed randomized
540-row single-image review CSV, and sealed condition key. Qwen/SAM results may
corroborate or disagree, but do not silently override the human record.

`cyan_initializer_ablation` is a one-variable scientific ablation with
`turquoise` selected before training and every other baseline setting fixed.
The selection came from qualitative folder-level review rather than a
completed blind-review ledger, so candidate win rates are unavailable. The historical
baseline configured `cyan`, which the cached tokenizer splits into two IDs;
the follow-up requires a true single-token initializer. This is a meaningful
initializer-semantics change and must not be described as an operational fix.
The matched 900-image run completed without black or invalid files, and the
user reported broad qualitative improvement. Because the review was not a
completed per-item blinded ledger, it supports the progression decision but
does not provide a numerical effect size. The new run also disables a
SafetyChecker that remained enabled in the frozen baseline, so raw black-image
counts are not a like-for-like model-quality metric.

`diagnostics_v1` changes generation conditions in ordered, isolated steps:
FP16 with the default SafetyChecker, FP16 with an explicitly disabled checker,
then checker-disabled FP32 plus finite learned-weight/pixel audits. These
conditions can diagnose black outputs but are not headline generation results.

`multiview_heldout_v1` changes both data volume and split structure. Its three
folds hold out different subject-color matchings, so any later result belongs
to a new multiview study and cannot be merged with the single-view baseline.
The fixed-neutral renderer now provides 180 validated views and image-only fold
staging, but no fold training or held-out metric is available. The rendered
study changes data volume, view distribution, and split structure relative to
the single-view baseline. A factor-aware loss would change the objective and
requires a separate method comparison; natural multi-object evaluation changes
the domain. Both remain conditional and have no result claim.
