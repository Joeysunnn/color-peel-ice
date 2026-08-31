# Reproduction commands

Statuses below describe this handoff; a command shown here is not evidence that
the corresponding model run succeeded.

## Baseline handoff verification — completed

```powershell
cd 'D:\UserFiles\Desktop\sr\color peel'
D:\anaconda3\python.exe -B -m pytest -q
D:\anaconda3\python.exe -B -m unittest discover -s tests -v
git diff --check
```

Final isolated per-file verification: `46 passed`, plus compile, JSON, and
diff checks. Runtime evidence is recorded separately in immutable server run
directories; a test result alone is never used as model-run evidence.

## GitHub push — completed

```powershell
git push --set-upstream origin repro/2026-08-21-colorpeel-clevr
```

`origin` is the user fork; `upstream` is the official repository.

## Server checkout from GitHub — completed and fast-forwarded

```bash
git clone --branch repro/2026-08-21-colorpeel-clevr --single-branch \
  https://github.com/Joeysunnn/color-peel-ice.git \
  /home/r12user5/Documents/Jiawei/colorpeel
git -C /home/r12user5/Documents/Jiawei/colorpeel status --short
git -C /home/r12user5/Documents/Jiawei/colorpeel rev-parse HEAD
```

The initial clone was later updated exclusively with `git pull --ff-only`.
Training launched from a clean checkout at
`c8c874d00318ae7c1df2265c8627787d316a1ce3`.

## Environment — completed and frozen

```bash
cd /home/r12user5/Documents/Jiawei/colorpeel
CONDA_BIN=/home/r12user5/miniforge3/bin/conda \
  bash scripts/setup/setup_colorpeel017.sh
```

Observed environment: Python 3.10.21, PyTorch 1.13.1+cu117, torchvision
0.14.1+cu117, official Diffusers v0.17.0, Transformers 4.30.2, Accelerate
0.20.3, and huggingface-hub 0.15.1. `pip check` passed. The full freeze is at
`/home/r12user5/Documents/Jiawei/colorpeel-runs/environment/colorpeel017-pip-freeze.txt`.

## Data audit and staging — completed

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

## Real training smoke 1 — passed

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

Evidence run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130559__clevr_subject_color_3x3__smoke2-first-two__c8c874d__42`.

## Real training smoke 2 — passed

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
ordinary-vocabulary drift. These observations passed in:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130706__clevr_subject_color_3x3__smoke9-full-grid__c8c874d__42`.
The validator reads `checkpoints/training_metrics.jsonl` and
`checkpoints/embedding_update_audit.json`; a missing or invalid observation
artifact prevents a smoke pass.

## Full 1500-step training — completed after both smoke passes

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__baseline__${COMMIT:0:7}__42"
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/baseline.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
```

Evidence run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130816__clevr_subject_color_3x3__baseline__c8c874d__42`.
Its manifest succeeded with return code 0 and 1500 finite metric rows.

## Generation — completed

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

Completed run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-132122__clevr_subject_color_3x3__generate__c8c874d__42`.
Its manifest succeeded with return code 0; all 900 images passed full decode,
RGB, and size validation.

## Independent Grounded-SAM stage — completed after one compatibility rerun

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

The first real run at `c8c874d` preserved 600 explicit failures caused by the
pinned Transformers 4.48.1 keyword difference. After the one-keyword `b059bd5`
patch was pushed and server-fast-forwarded, the completed runs were:

- segmentation:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-150928__clevr_subject_color_3x3__segment_grounded_sam__b059bd5__42`
- color scoring:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-151349__clevr_subject_color_3x3__score_color__b059bd5__42`

## Independent Qwen3-VL stage — completed

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

Completed runs:

- Qwen predictions:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-151546__clevr_subject_color_3x3__predict_qwen__b059bd5__42`
- CLEVR scoring:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-152105__clevr_subject_color_3x3__score_clevr__b059bd5__42`

## Evidence boundary

Before a real run: require clean Git provenance, tracked config/manifest, data
audit, environment/model provenance, launcher preflight, and observation-output
readiness. After a run: require immutable run manifest, command, environment,
stdout, exit code, observation artifacts, checkpoints and reload evidence.
Missing outputs remain `pending`; do not infer them from a process start or a
dry-run. In this execution both external stages have their own run manifests,
model provenance, per-item outputs, and explicit failure rows.
Generation and training use `colorpeel017`; Grounded-SAM and color scoring use
the existing `ice` environment; Qwen3-VL and CLEVR scoring use the existing
`ice-vlm` environment. The launcher always uses its current Python interpreter
and does not switch environments. Do not modify either existing ICE
environment. Both external model stages use `local_files_only`; a missing cache
must produce complete failure JSONL and a nonzero exit, never a model download
or silent substitution.

## Diagnosis-first provenance — reference only

The commands below are the approved follow-up workflow. Their presence is not
evidence that they ran. Start from the frozen source artifacts:

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_BASELINE_RUN="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/20260822-130816__clevr_subject_color_3x3__baseline__c8c874d__42"
export COLORPEEL_CHECKPOINT_DIR="$COLORPEEL_BASELINE_RUN/checkpoints"
export COLORPEEL_GENERATION_DIR="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/20260822-132122__clevr_subject_color_3x3__generate__c8c874d__42/inference"
git rev-parse HEAD
git status --short
```

Expected source provenance is training commit
`c8c874d00318ae7c1df2265c8627787d316a1ce3`. Do not resume the source run,
write into either source directory, or replace any source image.

## Tokenizer diagnosis — completed read-only check

```bash
conda run -n colorpeel017 python -c 'from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer", local_files_only=True); words=("cube","sphere","cylinder","red","cyan","gray","aqua","teal","turquoise"); print({w:t.encode(w, add_special_tokens=False) for w in words})'
```

Observed server output: `cube [11353]`, `sphere [6987]`, `cylinder [22092]`,
`red [736]`, `cyan [1470, 550]`, `gray [7048]`, `aqua [18613]`,
`teal [22821]`, and `turquoise [19899]`.

## Black-image SafetyChecker diagnosis — pending

Run the ordered stages only on the original black IDs. Each stage writes to a
new directory. If a stage has no still-black IDs, stop; do not force later
stages.

```bash
DIAG_ROOT="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/diagnostics_v1"
CUDA_VISIBLE_DEVICES=3 conda run -n colorpeel017 python \
  scripts/methods/colorpeel_ice/rerun_black_images.py \
  --manifest "$COLORPEEL_GENERATION_DIR/generation_manifest.jsonl" \
  --image-dir "$COLORPEEL_GENERATION_DIR" \
  --output-dir "$DIAG_ROOT/safety_flag" \
  --status-path "$DIAG_ROOT/safety_flag/rerun_status.jsonl" \
  --model-dir "$COLORPEEL_CHECKPOINT_DIR" \
  --device cuda:0 \
  --diagnostic-stage safety_flag \
  --dtype float16
```

Review `nsfw_content_detected` and the pixel audit before explicitly
acknowledging the checker-disabled stage:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n colorpeel017 python \
  scripts/methods/colorpeel_ice/rerun_black_images.py \
  --manifest "$COLORPEEL_GENERATION_DIR/generation_manifest.jsonl" \
  --image-dir "$COLORPEEL_GENERATION_DIR" \
  --output-dir "$DIAG_ROOT/disable_safety" \
  --status-path "$DIAG_ROOT/disable_safety/rerun_status.jsonl" \
  --prior-status "$DIAG_ROOT/safety_flag/rerun_status.jsonl" \
  --model-dir "$COLORPEEL_CHECKPOINT_DIR" \
  --device cuda:0 \
  --diagnostic-stage disable_safety \
  --dtype float16 \
  --disable-safety-checker \
  --acknowledge-safety-risk
```

Only IDs still black proceed to the finite FP32 diagnostic:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n colorpeel017 python \
  scripts/methods/colorpeel_ice/rerun_black_images.py \
  --manifest "$COLORPEEL_GENERATION_DIR/generation_manifest.jsonl" \
  --image-dir "$COLORPEEL_GENERATION_DIR" \
  --output-dir "$DIAG_ROOT/fp32_finite" \
  --status-path "$DIAG_ROOT/fp32_finite/rerun_status.jsonl" \
  --prior-status "$DIAG_ROOT/disable_safety/rerun_status.jsonl" \
  --model-dir "$COLORPEEL_CHECKPOINT_DIR" \
  --device cuda:0 \
  --diagnostic-stage fp32_finite \
  --dtype float32 \
  --disable-safety-checker \
  --acknowledge-safety-risk
```

Checker-disabled outputs are safety-sensitive diagnostics only. They must not
replace baseline images or be merged into baseline scores.

## Paired cyan diagnosis and review — completed generation

The diagnostic is fixed at ten nouns × seeds 42–44 × two prompt families. It
contains 300 trained-K/V rows (learned `<c2*>` plus four literal candidates)
and 240 vanilla-SD literal rows, for 540 images and 540 randomized single-image
blind-review rows:

```bash
CYAN_DIAG="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/diagnostics_v1/cyan_controls"
CUDA_VISIBLE_DEVICES=3 conda run -n colorpeel017 python \
  scripts/methods/colorpeel_ice/generate_cyan_diagnostic.py \
  --output-dir "$CYAN_DIAG" \
  --model-dir "$COLORPEEL_CHECKPOINT_DIR" \
  --device cuda:0 \
  --dtype float16

conda run -n colorpeel017 python \
  scripts/methods/colorpeel_ice/build_human_review.py \
  --manifest "$CYAN_DIAG/cyan_diagnostic_manifest.jsonl" \
  --image-dir "$CYAN_DIAG" \
  --blinded-image-dir "$CYAN_DIAG/human_review/images" \
  --review-csv "$CYAN_DIAG/human_review/review.csv" \
  --key-csv "$CYAN_DIAG/human_review/condition_key.csv" \
  --random-seed 20260822
```

The server run completed 540/540 status rows. The generated randomized review
packet is retained, but the initializer decision was made by qualitative
inspection of the named condition folders rather than a completed blind CSV;
no blind win rate is claimed.

Both `generate_cyan_diagnostic.py` and the subject diagnostic below keep the
default SafetyChecker enabled. A checker-disabled diagnostic requires the two
flags `--disable-safety-checker --acknowledge-safety-risk` together; never pass
only one flag, and never merge those outputs into the baseline.

## Subject diagnostic — completed generation

This fixed protocol creates 75 trained-K/V images: three shapes × seeds 42–46
× five conditions (learned subject-only, natural subject-only, and learned
subject paired with literal red, cyan, or gray):

```bash
SUBJECT_DIAG="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/diagnostics_v1/subject_controls"
CUDA_VISIBLE_DEVICES=3 conda run -n colorpeel017 python \
  scripts/methods/colorpeel_ice/generate_subject_diagnostic.py \
  --output-dir "$SUBJECT_DIAG" \
  --model-dir "$COLORPEEL_CHECKPOINT_DIR" \
  --device cuda:0 \
  --dtype float16
```

The real command writes `subject_diagnostic_manifest.jsonl`, 75 images, and
`subject_diagnostic_status.jsonl`. The server run completed 75/75 status rows.
To inspect the manifest without loading the model or generating images, append
`--dry-run` and omit `--model-dir`.

## Turquoise initializer ablation — selected, not run

Tracked config:
`experiments/clevr_subject_color_3x3/configs/train_cyan_initializer.yaml`.
Human review selected exactly `turquoise`. Run the two dedicated smoke configs
before the 1500-step config. Do not run the other candidates as a sweep.

```bash
export COLORPEEL_CONCEPTS_LIST="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/train_assets/concepts.json"
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__cyan_initializer_ablation__${COMMIT:0:7}__42"
CUDA_VISIBLE_DEVICES=3 conda run -n colorpeel017 python \
  scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/train_cyan_initializer.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID" \
  --dry-run
```

Run the two independent real smokes and validate their evidence:

```bash
for SPEC in "smoke_turquoise_2step:smoke2-turquoise-first-two" \
            "smoke_turquoise_9step:smoke9-turquoise-full-grid"; do
  CONFIG_NAME="${SPEC%%:*}"
  VARIANT="${SPEC#*:}"
  RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__${VARIANT}__${COMMIT:0:7}__42"
  RUN_DIR="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID"
  CUDA_VISIBLE_DEVICES=3 conda run -n colorpeel017 python \
    scripts/launch/colorpeel_run.py \
    --config "experiments/clevr_subject_color_3x3/configs/${CONFIG_NAME}.yaml" \
    --run-dir "$RUN_DIR"
  conda run -n colorpeel017 python src/train/training_audit.py validate \
    --config "experiments/clevr_subject_color_3x3/configs/${CONFIG_NAME}.yaml" \
    --run-dir "$RUN_DIR"
done
```

Only after both validators return `passed`, create a fresh run ID with variant
`cyan_initializer_ablation` and execute the same launcher command without
`--dry-run`. The tracked 1500-step config directly locks `turquoise`; no other
training argument changes.

Executed turquoise evidence runs:

- 2-step: `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-233300__clevr_subject_color_3x3__smoke2-turquoise-first-two__0959d1e__42`
- 9-step: `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-233340__clevr_subject_color_3x3__smoke9-turquoise-full-grid__0959d1e__42`
- 1500-step: `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-233433__clevr_subject_color_3x3__cyan_initializer_ablation__0959d1e__42`

Both smoke validators passed. The full manifest succeeded with return code 0
and 1500 finite metric rows.

## Multiview held-out protocol — locked renderer, execution pending

The renderer is tracked, but its runtime result must be established separately.
Plan requests with the real adapter:

```bash
python src/methods/colorpeel_ice/prepare_clevr_multiview.py plan \
  --output-dir "$RENDER_RUN/protocol" \
  --renderer scripts/methods/colorpeel_ice/render_clevr_multiview.py
```

Blender must be the verified 4.2.11 archive, run only with GPU 3. The first
command is a one-view smoke in an isolated output root. `--python-exit-code 1`
is required so Blender converts script exceptions into a failing process:

```bash
export CUDA_VISIBLE_DEVICES=3
BLENDER=/home/r12user5/Documents/Jiawei/tools/blender-4.2.11-linux-x64/blender
ASSETS=/home/r12user5/Documents/Jiawei/papers/CLEVER/image_generation/data

"$BLENDER" --background --python-exit-code 1 \
  --python scripts/methods/colorpeel_ice/render_clevr_multiview.py -- \
  --cycles-device CUDA --cycles-print-stats \
  --requests "$RENDER_RUN/protocol/render_requests.jsonl" \
  --profile experiments/clevr_subject_color_3x3/configs/multiview_render.json \
  --output-root "$SMOKE_RUN/rendered" \
  --properties-json "$ASSETS/properties.json" \
  --base-scene-blendfile "$ASSETS/base_scene.blend" \
  --shape-dir "$ASSETS/shapes" --material-dir "$ASSETS/materials" --limit 1
```

After inspecting smoke evidence, use a fresh full output root and the same
command without `--limit`. An interrupted full run may be continued only with
explicit `--resume`; all inputs, fingerprints, finalized files, and hashes must
still agree. Once 180 views complete, realize and stage the folds:

```bash
python src/methods/colorpeel_ice/prepare_clevr_multiview.py realize \
  --render-root "$RENDER_RUN/rendered" \
  --render-manifest "$RENDER_RUN/rendered/renderer_realization.jsonl" \
  --output-dir "$RENDER_RUN/prepared"
```

`realize` conditionally derives nine configs: folds A/B/C × seeds 42/43/44,
named `train_config_seed42.json`, `train_config_seed43.json`, and
`train_config_seed44.json` below each fold. Their variants are
`multiview_fold_{a|b|c}_seed{42|43|44}`. No renderer output, realized manifest,
or derived config is claimed until runtime evidence is recorded. Even after
successful staging, do not launch any fold smoke or 1500-step training before
the user reviews `multiview_human_review.csv` and
`multiview_contact_sheet.png`.

There is no approved command/config for a factor-aware loss or natural
multi-object evaluation. Both remain conditional; do not improvise an entry
point or report an output.

## Multiview renderer v2 orbit smoke — no Fold training

v2 must be selected explicitly; the default remains v1:

```bash
python src/methods/colorpeel_ice/prepare_clevr_multiview.py \
  --protocol experiments/clevr_subject_color_3x3/manifests/clevr_multiview_protocol_v2.json \
  plan --output-dir "$RENDER_V2_RUN/protocol" \
  --renderer scripts/methods/colorpeel_ice/render_clevr_multiview.py

export CUDA_VISIBLE_DEVICES=3
"$BLENDER" --background --python-exit-code 1 \
  --python scripts/methods/colorpeel_ice/render_clevr_multiview.py -- \
  --cycles-device CUDA --cycles-print-stats \
  --requests "$RENDER_V2_RUN/protocol/render_requests.jsonl" \
  --profile experiments/clevr_subject_color_3x3/configs/multiview_render_v2.json \
  --output-root "$SMOKE_V2_RUN/rendered" \
  --properties-json "$ASSETS/properties.json" \
  --base-scene-blendfile "$ASSETS/base_scene.blend" \
  --shape-dir "$ASSETS/shapes" --material-dir "$ASSETS/materials" --limit 1
```

Use a separate output root for every profile. Explicit `--resume` is allowed
only with the same v2 request/profile/asset contract. This command is a real
runtime smoke, not a realization and not authorization to start Fold training.

The accepted smoke was followed by this fresh full-run protocol at commit
`f7bc52d` (the renderer command was identical to the v2 command above except
for the new full output root and omission of `--limit`):

```bash
FULL_V2_RUN=/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260823-200500__clevr_subject_color_3x3__multiview-render-v2__f7bc52d__420000

python src/methods/colorpeel_ice/prepare_clevr_multiview.py \
  --protocol experiments/clevr_subject_color_3x3/manifests/clevr_multiview_protocol_v2.json \
  plan --output-dir "$FULL_V2_RUN/protocol" \
  --renderer scripts/methods/colorpeel_ice/render_clevr_multiview.py

CUDA_VISIBLE_DEVICES=3 "$BLENDER" --background --python-exit-code 1 \
  --python scripts/methods/colorpeel_ice/render_clevr_multiview.py -- \
  --cycles-device CUDA --cycles-print-stats \
  --requests "$FULL_V2_RUN/protocol/render_requests.jsonl" \
  --profile experiments/clevr_subject_color_3x3/configs/multiview_render_v2.json \
  --output-root "$FULL_V2_RUN/rendered" \
  --properties-json "$ASSETS/properties.json" \
  --base-scene-blendfile "$ASSETS/base_scene.blend" \
  --shape-dir "$ASSETS/shapes" --material-dir "$ASSETS/materials"

python src/methods/colorpeel_ice/prepare_clevr_multiview.py \
  --protocol experiments/clevr_subject_color_3x3/manifests/clevr_multiview_protocol_v2.json \
  realize --render-root "$FULL_V2_RUN/rendered" \
  --render-manifest "$FULL_V2_RUN/rendered/renderer_realization.jsonl" \
  --output-dir "$FULL_V2_RUN/prepared_human_review"
```

The renderer and realization both succeeded. No command from any staged Fold
config was executed.

After explicit human approval, all nine Fold configs passed `--dry-run` and
were launched sequentially with this command shape at commit `9cf1446`:

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config <prepared_human_review/folds/fold_{a,b,c}/train_config_seed{42,43,44}.json> \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/<immutable-run-id>"
```

The detached queue is serial on GPU 3 and stops at the first failure. It uses
no resume flag, scientific-parameter override, or concurrent Fold process.

## Multiview complete-bundle held-out evaluation

After all nine Fold runs succeed, validate them and derive nine immutable
generation configs plus one bundle config:

```bash
conda run -n colorpeel017 python -m src.methods.colorpeel_ice.prepare_multiview_evaluation \
  --training-root "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3" \
  --training-commit 9cf1446704ebf0fbc141e0fa28657f81204c8ac0 \
  --evaluation-protocol experiments/clevr_subject_color_3x3/manifests/clevr_multiview_heldout_eval_v1.json \
  --output-dir "$PLAN_ROOT"
```

Launch the nine generated configs serially on GPU 3 with the standard launcher.
Each run creates 180 complete-bundle images: nine cells × generation seeds
42–61. If interrupted, use launcher `--resume` against the same run directory;
the original commit, config, command, image hashes, model fingerprint, and
protocol fingerprint must still match.

```bash
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config "$PLAN_ROOT/generation_configs/fold_a_train42.json" \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/<immutable-run-id>"
```

Only after all nine runs succeed, execute the generated `bundle_config.json`
through the launcher. The bundle must contain exactly 1620 valid rows and emits
the randomized human ledger, nine contact sheets, and a gated Qwen config.
Human review remains primary. Qwen/scoring must not run before that review gate.

## Material renderer gate

```bash
conda run -n colorpeel017 python -m src.methods.colorpeel_ice.prepare_clevr_multiview_material plan \
  --output-dir "$MATERIAL_PLAN" \
  --renderer scripts/methods/colorpeel_ice/render_clevr_multiview.py
```

Run Blender with `--limit 2` first; request ordering guarantees the first two
items are the paired metal/rubber view for cube-red at seed 420000. After that
real smoke passes, render all 360 requests in a new output root. Realization
requires the accepted v2 render root and manifest. The completed run was
realized without rerendering using the approved decoded-pixel v2 gate:

```bash
RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260826-104500__clevr_subject_color_material_3x3x2__multiview-render-v3-material__d1f4282__420000
V2_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260823-200500__clevr_subject_color_3x3__multiview-render-v2__f7bc52d__420000
conda run -n colorpeel017 python -m src.methods.colorpeel_ice.prepare_clevr_multiview_material realize \
  --render-root "$RUN_ROOT/rendered" \
  --render-manifest "$RUN_ROOT/rendered/renderer_realization.jsonl" \
  --v2-render-root "$V2_ROOT/rendered" \
  --v2-render-manifest "$V2_ROOT/rendered/renderer_realization.jsonl" \
  --output-dir "$RUN_ROOT/prepared_human_review_v2_gate"
```

This command generated review/staging artifacts only. No generated training
config was executed.

## Material evaluator calibration

Set the accepted v3 realization and completed campaign bundle, then run the
three immutable stages through the standard launcher:

```bash
export COLORPEEL_MATERIAL_REFERENCE_ROOT=<prepared_human_review_v2_gate>
export COLORPEEL_MATERIAL_BUNDLE_RUN=<completed-3240-image-bundle-run>
conda run -n colorpeel017 python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_material_3x3x2/configs/prepare_material_calibration.json \
  --run-dir <immutable-prepare-run>

export COLORPEEL_MATERIAL_CALIBRATION_PREP_RUN=<immutable-prepare-run>
conda run -n ice-vlm python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_material_3x3x2/configs/predict_qwen_material_reference.json \
  --run-dir <immutable-qwen-reference-run>

export COLORPEEL_MATERIAL_REFERENCE_QWEN_RUN=<immutable-qwen-reference-run>
conda run -n ice-vlm python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_material_3x3x2/configs/score_material_reference.json \
  --run-dir <immutable-reference-score-run>
```

The preparation stage also creates 162 blinded A/B pair images, a blank review
CSV, a sealed key and nine contact sheets. Stop at the human pair-review gate.
