# Reproduction commands

Statuses below describe this handoff; a command shown here is not evidence that
the corresponding model run succeeded.

## Local verification — completed after current changes

```powershell
cd 'D:\UserFiles\Desktop\sr\color peel'
D:\anaconda3\python.exe -B -m pytest -q
D:\anaconda3\python.exe -B -m unittest discover -s tests -v
git diff --check
```

Observed after the current smoke-observation and independent-stage changes:
isolated pytest suite `44 passed`. This is code/config verification only; no
server model, training, generation, segmentation, prediction, or metric run is
implied.

## GitHub push — completed

```powershell
git push --set-upstream origin repro/2026-08-21-colorpeel-clevr
```

`origin` is the user fork; `upstream` is the official repository.

## Server checkout from GitHub — completed for implementation commit

```bash
git clone --branch repro/2026-08-21-colorpeel-clevr --single-branch \
  https://github.com/Joeysunnn/color-peel-ice.git \
  /home/r12user5/Documents/Jiawei/colorpeel
git -C /home/r12user5/Documents/Jiawei/colorpeel status --short
git -C /home/r12user5/Documents/Jiawei/colorpeel rev-parse HEAD
```

Verified clean at `41d752a9d8e8b3a5ab711db90990ab28e4f58000`.

## Environment — not run after rollback

```bash
cd /home/r12user5/Documents/Jiawei/colorpeel
CONDA_BIN=/home/r12user5/miniforge3/bin/conda \
  bash scripts/setup/setup_colorpeel017.sh
```

The previous temporary environment was intentionally removed. Recreate it before
the tracked data/preflight steps; literal official AdamW behavior is now the
locked policy and is no longer a setup blocker.

## Data audit and staging — not run with the tracked checkout

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
DATASET=/home/r12user5/Documents/Jiawei/papers/ICE/datasets/clevr_basic_neutral_stage1_gt
STAGING="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/train_assets"

conda run -n colorpeel017 python \
  src/methods/colorpeel_ice/prepare_clevr_3x3.py \
  --dataset-root "$DATASET" --output-dir "$STAGING" --dry-run
conda run -n colorpeel017 python \
  src/methods/colorpeel_ice/prepare_clevr_3x3.py \
  --dataset-root "$DATASET" --output-dir "$STAGING"
```

All three training configs use the same ordered full-grid concepts list:

```bash
export COLORPEEL_CONCEPTS_LIST="$STAGING/concepts.json"
```

The training loader is intentionally non-shuffled. Therefore the two-step
smoke observes the first two locked manifest rows, while the nine-step smoke
observes every row once; no derived concepts file is created.

## Tracked launcher dry-runs — preflight only, not training smokes

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__smoke2-first-two__${COMMIT:0:7}__42"
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/smoke_2step.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID" \
  --dry-run
```

Repeat with `smoke_9step.yaml` and a run ID whose variant is
`smoke9-full-grid`. Every dry-run directory is consumed; use a new run ID for
real execution. A successful preflight is not either of the two real smokes.

## Real training smoke 1 — two steps, first two samples, not run

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__smoke2-first-two__${COMMIT:0:7}__42"
SMOKE2_RUN="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/smoke_2step.yaml \
  --run-dir "$SMOKE2_RUN"
conda run -n colorpeel017 python src/train/training_audit.py validate \
  --config experiments/clevr_subject_color_3x3/configs/smoke_2step.yaml \
  --run-dir "$SMOKE2_RUN"
```

Expected exposure counts are `<s1*>:2`, `<c1*>:1`, `<c2*>:1`, and zero for
`<s2*>`, `<s3*>`, `<c3*>`. Only exposed tokens require a nonzero-gradient
observation and nonzero final delta. Unseen tokens are not failures. Record
ordinary-vocabulary AdamW drift but do not restore it or use it as pass/fail.

## Real training smoke 2 — nine steps, full grid, not run

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__smoke9-full-grid__${COMMIT:0:7}__42"
SMOKE9_RUN="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/smoke_9step.yaml \
  --run-dir "$SMOKE9_RUN"
conda run -n colorpeel017 python src/train/training_audit.py validate \
  --config experiments/clevr_subject_color_3x3/configs/smoke_9step.yaml \
  --run-dir "$SMOKE9_RUN"
```

Each modifier token must have exposure count exactly 3, at least one nonzero
gradient observation, and nonzero final embedding delta. Both smoke runs must
also record finite losses, exit code, saved artifacts, reload outcome, and
ordinary-vocabulary drift. These observations are currently `not_run`.
The validator reads `checkpoints/training_metrics.jsonl` and
`checkpoints/embedding_update_audit.json`; a missing or invalid observation
artifact prevents a smoke pass.

## Full 1500-step training — only after both smoke records are complete

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__baseline__${COMMIT:0:7}__42"
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/baseline.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
```

## Generation — tracked independent run, not started

```bash
TRAIN_RUN="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/<completed-baseline-run-id>"
export COLORPEEL_CHECKPOINT_DIR="$TRAIN_RUN/checkpoints"
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__generate__${COMMIT:0:7}__42"
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/generate.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
```

Replace the placeholder with the reviewed completed baseline run. Generation
does not run Grounded-SAM or Qwen3-VL.

## Independent Grounded-SAM stage — implemented/configured, not run

Grounded-SAM consumes only the transfer rows from the generation manifest and
writes explicit per-item status plus masks mirroring each generated
`image_path` in its own immutable run directory:

```bash
GENERATE_RUN="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/<completed-generate-run-id>"
export COLORPEEL_GENERATION_DIR="$GENERATE_RUN/inference"
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__segment_grounded_sam__${COMMIT:0:7}__42"
/home/r12user5/miniforge3/envs/ice/bin/python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/segment.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
```

After that stage completes, run color scoring:

```bash
SEGMENT_RUN="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/<completed-segment-run-id>"
export COLORPEEL_MASK_DIR="$SEGMENT_RUN/evaluation/masks"
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__score_color__${COMMIT:0:7}__42"
/home/r12user5/miniforge3/envs/ice/bin/python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/score_color.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
```

## Independent Qwen3-VL stage — implemented/configured, not run

Qwen3-VL consumes the 300 non-transfer rows with deterministic decoding and
`max_new_tokens=128`, writing one success/failure JSON row per image:

```bash
export COLORPEEL_GENERATION_DIR="$GENERATE_RUN/inference"
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__predict_qwen__${COMMIT:0:7}__42"
/home/r12user5/miniforge3/envs/ice-vlm/bin/python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/predict_qwen.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
```

After that stage completes, score its predictions:

```bash
QWEN_RUN="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/<completed-qwen-run-id>"
export COLORPEEL_QWEN_PREDICTIONS="$QWEN_RUN/evaluation/qwen_predictions.jsonl"
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__score_clevr__${COMMIT:0:7}__42"
/home/r12user5/miniforge3/envs/ice-vlm/bin/python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/score_clevr.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
```

## Evidence boundary

Before a real run: require clean Git provenance, tracked config/manifest, data
audit, environment/model provenance, launcher preflight, and observation-output
readiness. After a run: require immutable run manifest, command, environment,
stdout, exit code, observation artifacts, checkpoints and reload evidence.
Missing outputs remain `pending`; do not infer them from a process start or a
dry-run. Grounded-SAM and Qwen3-VL remain separate `not_run` stages until their
own run manifests, model provenance, per-item outputs, and failure rows exist.
Generation and training use `colorpeel017`; Grounded-SAM and color scoring use
the existing `ice` environment; Qwen3-VL and CLEVR scoring use the existing
`ice-vlm` environment. The launcher always uses its current Python interpreter
and does not switch environments. Do not modify either existing ICE
environment. Both external model stages use `local_files_only`; a missing cache
must produce complete failure JSONL and a nonzero exit, never a model download
or silent substitution.
