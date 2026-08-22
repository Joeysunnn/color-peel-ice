# Patch record

## Verified implementation history

- Historical baseline branch: `repro/2026-08-21-colorpeel-clevr`
- Current diagnosis-first branch: `repro/2026-08-22-colorpeel-diagnostics`
- Baseline: `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- Commit: `41d752a9d8e8b3a5ab711db90990ab28e4f58000`
- Message: `repro: add ColorPeel CLEVR 3x3 workflow`
- Pre-run workflow: `6adca6dab9ad177c58a99b6ab26662cc92e8c140`
- Accelerate compatibility: `c8c874d00318ae7c1df2265c8627787d316a1ce3`
- Grounding DINO compatibility: `b059bd5e92cf1994581d8600111d3ed5830dc7d5`
- Highest verified risk: final modifier-token gradient-boundary correction
- README fidelity: adapted dataset/protocol; core ColorPeel training settings
  preserved; no training result claimed.

## Patch groups

### Data and study definition

- `src/methods/colorpeel_ice/prepare_clevr_3x3.py`
- `experiments/clevr_subject_color_3x3/`
- `tests/methods/colorpeel_ice/test_prepare_clevr_3x3.py`

Locks the nine IDs, 3×3 token map, RGB values, metal material, image-only
staging, hashes, and GT-mask audit boundary. Synthetic local tests passed; the
tracked script also passed against the real 48-sample server root.

### Training correction and launcher

- `src/train/train_colorpeel.py`
- `src/train/token_gradient_utils.py`
- `scripts/methods/colorpeel_ice/train.sh`
- `scripts/launch/colorpeel_run.py`
- `tests/methods/colorpeel_ice/test_modifier_gradient_utils.py`
- `tests/test_experiment_runner.py`

The official loop omitted the last modifier ID when building its gradient
mask. The helper now preserves all six modifier rows and zeros all other
gradient rows. Local boundary and launcher tests passed. Literal official
AdamW decay of ordinary vocabulary values is intentionally allowed and must be
observed.

### Generation and evaluation adapters

- `scripts/methods/colorpeel_ice/generate.py`
- `scripts/methods/colorpeel_ice/score_clevr_predictions.py`
- `scripts/methods/colorpeel_ice/evaluate_color_metrics.py`
- corresponding tests under `tests/methods/colorpeel_ice/`

Loads all six tokens and Custom Diffusion weights, passes steps/CFG/generator
seeds, builds the 900-row manifest, emits CLEVR accuracy/contingency tables,
and computes color metrics from external masks. Local tests and every real
stage completed; per-stage success/failure counts are recorded in `LOG.md`.

### Current pre-run handoff additions

- `experiments/clevr_subject_color_3x3/configs/smoke_2step.yaml`
- `experiments/clevr_subject_color_3x3/configs/smoke_9step.yaml`
- `scripts/methods/colorpeel_ice/segment_grounded_sam.py`
- `scripts/methods/colorpeel_ice/predict_qwen.py`
- tracked generation, segmentation, Qwen prediction, and scorer configs
- training observation fields and audit/report template updates

These additions define the two real smoke contracts, literal official AdamW
drift policy, and independent evaluation stages. They were pushed in
`6adca6d`, deployed by `git pull --ff-only`, and the server test suite passed
44 tests before real execution.

### Accelerate 0.20.3 compatibility patch

- Commit: `c8c874d00318ae7c1df2265c8627787d316a1ce3`
- Scope: move `logging_dir` into `ProjectConfiguration`, where pinned
  Accelerate 0.20.3 accepts it.
- Evidence: the superseded first smoke stopped at zero steps; fresh two-step,
  nine-step, and full 1500-step runs subsequently succeeded.
- Scientific impact: none. Training data, losses, optimizer, trainable
  parameters, gradients, and update order were unchanged.

### Transformers 4.48.1 Grounding DINO compatibility patch

- Commit: `b059bd5e92cf1994581d8600111d3ed5830dc7d5`
- Scope: use the pinned processor's `box_threshold` keyword while preserving
  the locked numeric threshold 0.25.
- Evidence: the first stage retained 600 explicit `TypeError` failure rows;
  the fresh run produced 588 valid masks and 12 semantic/ratio failures.
- Scientific impact: none expected. Model IDs, thresholds, prompts, mask-ratio
  bounds, images, and metric implementation are unchanged.

### Workflow and evidence structure

- `doc/`, `literature/`, `environment/`, and root `README.md`

Adapts the prior project's structural conventions only: method/study split,
tracked definitions, evidence labels, GitHub-only deployment, and external run
artifacts. No ICE method implementation was copied.

The official `src/test/test.py` behavior was restored during method-directory
reorganization. Its only remaining byte-level difference is a final newline
added by `apply_patch`; no executable line changed.

## Verification

```text
server GitHub checkout: clean at c8c874d
pre-run handoff: server pytest 44 passed
compatibility regression suite: 14 passed locally
2-step smoke: passed
9-step smoke: passed
1500-step training: passed
generation: 900/900 valid
Grounded-SAM: 588/600 masks, 12 explicit failures
Qwen3-VL: 300/300 valid predictions
CLEVR grid: shape 94.44%, color 93.89%, joint 93.89%
```

## Explicit no-patch boundary

Do not add non-modifier embedding restoration. The tracked policy is
`literal_official_adamw_decay_allowed`: record ordinary-vocabulary drift, do
not restore it, and do not classify nonzero drift as failure. Both training
smokes passed under this policy. The independent Grounded-SAM/Qwen3-VL stages
remain separate evidence contracts and are not implied by training success.

## Diagnosis-first pre-run patch set

Status: included in the current pre-run handoff; no diagnosis, render,
follow-up training, or new evaluation result is claimed here. This section does
not assign a commit hash or claim server deployment.

- `experiments/clevr_subject_color_3x3/configs/train_cyan_initializer.yaml`
  declares the conditional `cyan_initializer_ablation`. It changes only the
  `<c2*>` initializer; `turquoise` is locked directly as the human-selected
  candidate so a runtime variable cannot substitute another word.
- `experiments/clevr_subject_color_3x3/configs/smoke_turquoise_2step.yaml` and
  `smoke_turquoise_9step.yaml` preserve the established smoke contracts while
  changing only the `<c2*>` initializer for the selected ablation.
- `src/train/initializer_token_utils.py` and its training call site reject
  multi-piece initializer words. This is scientifically meaningful for future
  runs because cached SD 1.4 tokenization splits `cyan` into `[1470, 550]`.
  The frozen baseline is not rewritten.
- `scripts/methods/colorpeel_ice/generate_cyan_diagnostic.py` and
  `build_human_review.py` define paired cyan controls and a blinded review
  packet. They create no result until 540 images and a completed 540-row review CSV
  exist.
- `scripts/methods/colorpeel_ice/generate_subject_diagnostic.py` defines the
  75-image trained-K/V subject control. It is diagnostic-only and does not
  make single-axis generation a training objective.
- `scripts/methods/colorpeel_ice/rerun_black_images.py` defines three ordered
  safety/precision diagnostics. Checker disablement is diagnostic-only,
  explicitly acknowledged, isolated from baseline outputs, and never made the
  default.
- `experiments/clevr_subject_color_3x3/manifests/clevr_multiview_protocol.json`,
  its schema, and `src/methods/colorpeel_ice/prepare_clevr_multiview.py` define
  a renderer-owned held-out protocol. No real renderer output or training is
  part of this patch evidence.
- Audit and experiment documents record baseline freeze, human-evidence
  priority, stage gates, exact commands, comparability limits, and pending
  status.

Highest follow-up scientific risk: changing initializer semantics for a new
training run. Highest safety-relevant diagnostic risk: disabling the Stable
Diffusion SafetyChecker. A factor-aware loss and natural multi-object
evaluation have no approved implementation/config and remain conditional.

Verified pre-run commit: `ca3d313c4d081bcdec5fda6979b05c4fde3415c0`
(`repro: add diagnosis-first ColorPeel controls`). Local and server suites each
passed 73 tests. Runtime evidence is stored outside Git under the diagnostics
run root recorded in `LOG.md`; generated images and learned weights remain
outside Git.
