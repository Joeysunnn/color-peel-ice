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
