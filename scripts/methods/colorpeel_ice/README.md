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
The unresolved AdamW weight-decay effect on non-modifier embedding rows must be
reviewed before a full run; see the study report and `repro_outputs/PATCHES.md`.

## Generation and scoring

```bash
python scripts/methods/colorpeel_ice/generate.py --help
python scripts/methods/colorpeel_ice/score_clevr_predictions.py --help
python scripts/methods/colorpeel_ice/evaluate_color_metrics.py --help
```

These entry points implement the locked 900-image generation manifest and
artifact-based scoring adapters. They do not claim that Grounded-SAM or
Qwen3-VL has been run.
