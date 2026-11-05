# Wood spoon: integrate soft TRELLIS alpha while retaining SAM identity

This bounded verification keeps the current Scheme A photograph, SAM instance,
model revisions, seed42, and pink hue0. It recomputes pinned BiRefNet on the full
native original RGB photograph, retains complete alpha components associated
with that SAM instance, jointly resizes RGBA using the historical path, recolors
only RGB, and samples one fresh mesh. It does not prepare S11 or train any branch.

The SAM segmentation from `7726b73` is intentionally frozen for this control.
All new input preparation and mesh generation must use the same clean Git commit.
The historical soft cutout is only read to compare the freshly inferred result;
it is never used as a fallback input. Cached model files are verified offline.

Production integration is in `single_image_pipeline.py`: original input receipt
→ `subject_alpha` → pink input → TRELLIS. The canonical SAM mask and original
photo fields remain unchanged for training, photo anchors, and C/M/T. Soft alpha
has separate receipt fields. Shared soft-alpha components or insufficient SAM
coverage fail visibly instead of clipping the matte along a hard SAM boundary.

After the relevant unit tests execute in the clean server checkout, use:

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT="$COLORPEEL_RUN_ROOT"
/home/r12user5/miniforge3/envs/ice/bin/python -B -m experiments.wood_spoon_soft_alpha_v1.run \
  --output "$COLORPEEL_RUN_ROOT/wood_spoon_soft_alpha_v1/<new-run-id>" --gpu 0
```

Inspect `input_comparison.json`, the semantic and conditioning receipts, and the
new clay/normal previews before drawing a quality conclusion. A completed run,
unchanged mask, or silhouette IoU is not user acceptance of mesh quality.

The completed seed42 integration verification is recorded in [RESULTS.md](RESULTS.md).

## Original-color control with the same new alpha

To isolate conditioning color, reuse `trellis_alpha/source_receipt.json` from
the completed run, before the pink stage. Use its clean producer checkout
`d5b9d15f845e13b2ef26e09e01481b46f5a52559`; check GPU availability and all source
locks first. Do not rerun segmentation, substitute the historical receipt,
change production defaults, or reuse an existing output directory.

```bash
PINK_RUN="$COLORPEEL_RUN_ROOT/wood_spoon_soft_alpha_v1/20261105__sam_soft_pink__d5b9d15__42"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python -u -B \
  -m experiments.subject_trellis_counterfactual_v1.run \
  --source-receipt "$PINK_RUN/trellis_alpha/source_receipt.json" \
  --output "$COLORPEEL_RUN_ROOT/wood_spoon_soft_alpha_v1/NEW_ORIGINAL/trellis" \
  --encoder /var/tmp/colorpeel-r12user5/assets/trellis2_v1/dino_timm_v2/model \
  --model-revisions experiments/subject_trellis_spoons_v1/model_revisions.json \
  --seed 42 --skip-glb
```

After generation completes, `experiments.subject_pink_input_repro_v1.compare`
can compare the actual conditioning pixels, model/sampler settings and all five
native arrays against the historical mesh and the pink control. Its output must
be a new JSON path outside the compared mesh directories. This does not train
or expand any branch dataset.

Completed original-color control: [ORIGINAL_RESULT.md](ORIGINAL_RESULT.md).
