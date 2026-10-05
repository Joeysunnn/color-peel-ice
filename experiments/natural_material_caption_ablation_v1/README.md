# Metal spoon caption-only ablation

This diagnostic uses the completed `natural_material_fullmaps_v1` metal-spoon
training images and masks by their original absolute paths. The only change to
the examples seen by the trainer is the caption: each sphere, cube, and
cylinder uses `a photo of an object made of <M*>`. The source Albedo, Roughness,
Metallic, render conditions, 72 examples, token-local K/V update, initializer,
optimizer, seed, and 5000-step schedule remain as in the source config.

On the server, after this script is present in a clean committed checkout:

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
SOURCE="$COLORPEEL_RUN_ROOT/natural_material_fullmaps_v1/run_20261004_fullmaps_threeway_lighting_v2"
STAGING="$COLORPEEL_RUN_ROOT/natural_material_caption_ablation_v1/staging_metal_spoon_generic_caption"
python experiments/natural_material_caption_ablation_v1/prepare.py \
  --source-run "$SOURCE" --output "$STAGING"
```

Inspect `audit.json`, then launch with the usual clean-worktree launcher. Give
the run directory a fresh timestamp and the checkout's current seven-character
Git commit. Example, replacing `YYYYMMDD-HHMMSS` and `COMMIT7`:

```bash
RUN="$COLORPEEL_RUN_ROOT/natural_material_caption_ablation_v1/YYYYMMDD-HHMMSS__natural_material_caption_ablation_v1__metal_spoon_generic_caption_token_local_kv_5000__COMMIT7__42"
python scripts/launch/colorpeel_run.py --config "$STAGING/train.json" --run-dir "$RUN" --dry-run
# Use a different fresh timestamp for the actual training run; dry-run creates its directory.
```

`prepare.py` refuses an existing output and verifies every original image and
mask hash before writing. `audit.json` records source/config hashes and the
permitted differences. Training is intentionally not launched by preparation.
