# ColorPeel-on-CLEVR method entry points

This directory contains the CLEVR adaptation entry points. The official
ColorPeel training implementation remains under `src/train/`; the only change
inside that implementation is the recorded modifier-token gradient-boundary
correction.

The fixed study definition, sample manifest, configs, and research record live
under `experiments/clevr_subject_color_3x3/`. Runtime images, checkpoints, and
logs belong below an external `$COLORPEEL_RUN_ROOT`, not in Git.

## Data preparation

```bash
python src/methods/colorpeel_ice/prepare_clevr_3x3.py \
  --dataset-root "$COLORPEEL_DATA_ROOT/clevr_basic_neutral_stage1_gt" \
  --output-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/train_assets"
```

The source GT masks are validated and hashed, but the isolated training
directories contain only `img.jpg` links. Masks do not enter the ColorPeel
training loss.

## Training

```bash
CONCEPTS_LIST="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/train_assets/concepts.json" \
OUT_DIR="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/<run_id>/checkpoints" \
CUDA_DEVICE=3 \
bash scripts/methods/colorpeel_ice/train.sh
```

The launcher uses the official SD 1.4, 512 resolution, batch size 1, 1500
steps, learning rate `1e-5`, CAA weight 0.2, and six shared modifier tokens.
The optimizer policy is `literal_official_adamw_decay_allowed`: ordinary
vocabulary embedding drift is measured and reported, but it is not restored
and is not a failure by itself. Run the independent two-step and nine-step
training smoke configs before the full run; launcher `--dry-run` is preflight,
not a smoke. Training writes `training_metrics.jsonl` and
`embedding_update_audit.json` for the smoke validator.

## Generation, external-model stages, and scoring

```bash
python scripts/methods/colorpeel_ice/generate.py --help
python scripts/methods/colorpeel_ice/segment_grounded_sam.py --help
python scripts/methods/colorpeel_ice/predict_qwen.py --help
python scripts/methods/colorpeel_ice/score_clevr_predictions.py --help
python scripts/methods/colorpeel_ice/evaluate_color_metrics.py --help
```

These are independent tracked stages:

- training and generation run with `colorpeel017`;
- Grounded-SAM and color scoring run with the existing
  `/home/r12user5/miniforge3/envs/ice/bin/python`;
- Qwen3-VL prediction and CLEVR scoring run with the existing
  `/home/r12user5/miniforge3/envs/ice-vlm/bin/python`.

The launcher uses the Python interpreter that invokes it and never switches
environments. Do not modify the two existing ICE environments. Grounded-SAM
and Qwen3-VL use `local_files_only`; cache absence produces explicit per-item
failure JSONL and a nonzero stage exit rather than downloading or substituting
a model. Entry-point availability does not claim that generation,
Grounded-SAM, Qwen3-VL, or either scorer has run.

## Diagnosis-first follow-up

The report-01 checkpoint is a read-only comparison anchor. Follow-up scripts
always write to new external run directories:

```bash
python scripts/methods/colorpeel_ice/rerun_black_images.py --help
python scripts/methods/colorpeel_ice/generate_cyan_diagnostic.py --help
python scripts/methods/colorpeel_ice/generate_subject_diagnostic.py --help
python scripts/methods/colorpeel_ice/build_human_review.py --help
```

`rerun_black_images.py` enforces the ordered stages `safety_flag`,
`disable_safety`, and `fp32_finite`. The first remains FP16 with the default
SafetyChecker. Later stages continue only IDs still black in the prior status;
checker-disabled stages require `--acknowledge-safety-risk`. Their outputs are
diagnostic and may not replace baseline images.

`generate_cyan_diagnostic.py` defines 540 controls: ten nouns, seeds 42–44,
two prompt families, 300 trained-K/V rows, and 240 vanilla-SD rows. The trained
rows include learned `<c2*>` and four literal candidates; vanilla rows include
the four literal candidates. `build_human_review.py` creates 540 randomized
single-image blind-review rows plus a separate key. A completed human review is
primary semantic evidence; script availability alone is not a result.

`generate_subject_diagnostic.py` defines 75 trained-K/V images: three shapes,
seeds 42–46, and five conditions per shape/seed—learned subject-only, natural
subject-only, and learned subject with literal red, cyan, or gray. It writes a
manifest in dry-run mode and per-image status during a real run. No subject
diagnostic run is currently claimed.

The cyan and subject generators default to the SafetyChecker enabled. Either
generator accepts checker disablement only with the paired flags
`--disable-safety-checker --acknowledge-safety-risk`; checker-disabled outputs
remain diagnostic-only.

The follow-up training config is
`experiments/clevr_subject_color_3x3/configs/train_cyan_initializer.yaml`.
It locks the reviewed choice `turquoise`. Only `<c2*>` initialization may
change; running the other candidates as an undeclared sweep is outside the
approved protocol. No ablation run is currently claimed.

## Controlled two-object stage

The versioned two-object protocol lives under
`experiments/clevr_two_object_subject_color_material/`. It keeps the accepted
eight shared subject/color/material tokens and creates two object-level
training records from each two-object RGB scene. Each record has one complete
token bundle and one matching GT instance mask; CAA itself is unchanged and
never mixes tokens from the two objects.

```bash
python src/methods/colorpeel_ice/prepare_clevr_two_object.py plan \
  --output-dir "$COLORPEEL_RUN_ROOT/two_object/plan" \
  --renderer scripts/methods/colorpeel_ice/render_clevr_two_object.py

blender --background --python-exit-code 1 \
  --python scripts/methods/colorpeel_ice/render_clevr_two_object.py -- \
  --cycles-device CUDA --cycles-print-stats \
  --requests "$COLORPEEL_RUN_ROOT/two_object/plan/render_requests.jsonl" \
  --profile experiments/clevr_two_object_subject_color_material/configs/multiview_render_v4_two_object.json \
  --output-root "$COLORPEEL_RUN_ROOT/two_object/render_smoke" \
  --properties-json "$CLEVR_ROOT/data/properties.json" \
  --base-scene-blendfile "$CLEVR_ROOT/data/base_scene.blend" \
  --shape-dir "$CLEVR_ROOT/data/shapes" \
  --material-dir "$CLEVR_ROOT/data/materials" \
  --limit 2
```

The two-image prefix is a real Blender smoke for the forward and swapped
orientations of the first pair. A successful smoke does not authorize
training: complete the 360-scene render, realization, contact sheets, and the
tracked human gate first.

After the authorized seed-42 training completes, `generate_two_object.py`
loads the same eight-token checkpoint and evaluates explicit left/right bundle
composition. The versioned protocol contains 360 seen-pair and 360 unseen-pair
images. `bundle_two_object_evaluation.py` validates all artifacts and creates
the human-review packet; it does not run Qwen or claim disentanglement.
