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
- `scripts/methods/colorpeel_ice/generate.py` now applies the already approved
  SafetyChecker policy to matched evaluation: disablement requires an explicit
  paired acknowledgement and both booleans are written into all 900 manifest
  rows. This does not alter the training checkpoint or frozen baseline images.
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

## Multiview preflight hardening

- `configs/multiview_base_turquoise.yaml` keeps all official training values
  and the literal official AdamW policy while locking the selected one-token
  `turquoise` initializer.
- `prepare_clevr_multiview.py` now rejects a wrong modifier/initializer map,
  protocol identity drift, nonempty output directories, contaminated staging,
  repeated rendered images, and absent camera/light/background variation.
- Fold evidence now separates held-out train-index views (48 per fold) from
  matched held-out audit-index views (12 per fold); GT masks remain excluded
  from all training concept directories.
- These are protocol-integrity changes. They do not change ColorPeel's model,
  loss, optimizer, gradients, data loader order, or checkpoint format.
- Verified code commit: `50b745a1da8c1418a360eac5b9180a4c7e0b36dc`.
  Local and server suites each passed 77 tests; server deployment was a clean
  GitHub-only fast-forward.

## Fixed-neutral renderer realization

- Added locked Blender 4.2.11/Cycles CUDA adapter and immutable
  `multiview_render_v1` profile. World/ground stay fixed; only camera and three
  named lights receive deterministic official-style jitter.
- Resume verifies request/profile/asset fingerprints, finalized file hashes,
  per-view records, and directory purity before skipping anything.
- Realization requires RGB decode, complementary binary object/background
  masks, finite bounded mask area, object visibility, fixed background,
  deterministic transform metadata, one V100 CUDA device, and unique images.
- Runtime-only fixes place locked Cycles flags after Blender's `--` and compare
  Blender float32 transforms with `1e-6` tolerance. Scientific parameters and
  ColorPeel training math are unchanged.
- Renderer commit `53a3a0e`; final validator/gate commit `de279ca`. Full run
  completed 180/180 with zero failures; no fold training was run.

## Versioned orbit renderer v2

- Added a separate immutable v2 profile/protocol/schema; v1 files and default
  dispatch remain unchanged and are guarded by historical SHA-256 tests.
- Replaced only camera XYZ translation jitter with deterministic spherical
  orbit sampling around the realized object center. The first three RNG draws
  remain camera-owned, so same-seed light offsets are identical to v1.
- Scene metadata and validator now record/recompute base/final pose, spherical
  offsets, object-center target, normalized quaternion, `-Z` alignment,
  Y-up roll, optical center, base constraint state, Cycles/GPU/profile/assets,
  and exact fixed background.
- Real Blender smoke required two v2-only runtime corrections: record/mute the
  base scene's `Track To` constraint before explicit look-at, and refresh the
  dependency graph after setting orbit location. No training or v1 render
  behavior changed.
- Implementation commits: `097ec27`, `28a51bb`, and `bb75930`. Real smoke
  completed five cube views and one cylinder view; full v2 realization and
  Fold training remain unrun.

## Multiview held-out campaign adapters

- Added an immutable 1620-row complete-bundle protocol and strict shared
  validator for the exact Fold/seed/cell/generation-seed matrix.
- Added generation provenance binding to the parent training manifest/config,
  final model artifacts, protocol file, per-image status, and SHA-256.
- Added a nine-run bundle stage that rejects incomplete, conflicting, duplicate,
  non-RGB, wrong-size, or hash-mismatched outputs before creating a legal
  campaign manifest, randomized human-review ledger, and contact sheets.
- Added launcher and Qwen resume paths. Resume is allowed only for the exact
  original run revision/config/command; Qwen persists one row at a time.
- Added a separate held-out scorer. The original baseline generator and scorer
  remain unchanged; no training math, loss, optimizer, mask boundary, checkpoint,
  or prompt template changed.

## Three-attribute material extension

- Added `multiview_render_v3_material` without modifying v1/v2 profile data or
  request schemas. V3 request validation adds `shape_color_index` and
  `material_token`; resume validates both.
- V3 hashes both `MyMetal.blend` and `Rubber.blend`, records the selected asset
  in scene metadata, and keeps material selection outside the RNG stream.
- Added a separate material prepare/realize path. It rejects realization unless
  paired camera/light metadata and decoded masks match. Accepted-v2 metal RGB
  uses versioned `decoded_pixel_equivalence_v2` (`mean <= 0.001`, changed-channel
  fraction `<= 0.001`; max is record-only); decoded masks remain exact and all
  raw hashes are retained as evidence. Historical v1 remains defined.
- Generalized only smoke artifact validation to derive modifier files from the
  training config. Historical six-token constants and two-axis evaluation
  adapters remain intact.
- Tightened the training preflight to require equal, unique modifier/initializer
  counts and newly added modifier tokens. Loss, optimizer, gradient masking,
  update order, checkpoint format, and ordinary-vocabulary AdamW behavior are
  unchanged.
