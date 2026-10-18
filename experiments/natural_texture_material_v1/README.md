# Phase 2: natural Texture and Material

This independent study continues the accepted Phase 1 SD1.5 experiment.
It uses the same mailbox, metal_spoon and wood_spoon natural inputs as
`natural_material_fullmaps_lora_v1`, following its frozen data provenance to
the original SuperMat outputs and reviewed material regions. The old study's
SD1.4/rank4/full-A-R-M definition and training schedule are not inherited.
Existing data, checkpoints, results and Subject/Color branches remain unchanged.

## Ordered delivery

1. Extract and display the natural source, reviewed region, original SuperMat
   Albedo/Roughness/Metallic, normalized grayscale Texture and scalar R/M.
2. Render and display the two independent counterfactual training datasets.
3. Train original token-local K/V and rank64/alpha64 token-local K/V for each
   source and branch, saving paired embeddings/adapters at500/750/1000 updates,
   then generate matched comparisons in separate step directories.

Only step1 is implemented by `extract.py`; it never renders or starts training.
All code is edited/tested locally, committed/pushed, then fetched and pulled
with `--ff-only` on the server before execution from a clean checkout.

## Reproduce the extraction preview

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export PYTHONDONTWRITEBYTECODE=1
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
SOURCE=$COLORPEEL_RUN_ROOT/natural_material_fullmaps_v1/run_20261004_fullmaps_threeway_lighting_v2
# NEW_EXTRACTION must be a new directory below COLORPEEL_OUTPUT_ROOT.
$PY -B -m experiments.natural_texture_material_v1.extract \
  --source-fullmaps-run "$SOURCE" --output "$NEW_EXTRACTION"
```

Reuse `prepare_maps.source_paths` to resolve each original source and reviewed
mask. Texture extraction calls Phase 1 `data.extract_texture`: linear-sRGB
Rec709 luminance, valid-region2/98 percentiles, output range[0.25,0.75]. Crop
the reviewed-region bounding box and fill only invalid pixels with the existing
TELEA radius3 rule; preserve every valid grayscale pixel. Record the crop and
filled fraction. This estimates an image-space texture patch, not recovered
physical UV coordinates or ground-truth intrinsic Albedo.

Material extraction reuses `material_pipeline.map_values`: separate original
8-bit R/M maps divided by255, median over the same reviewed region, with
MAD/IQR/quantile/hash records. Scalar estimates are checked against the frozen
canonical values. Retain the wood source's existing roughness/gloss conflict;
do not adjust R/M to match a semantic label. No IID, learned extraction model,
extra disentanglement loss or architecture is introduced.

## Subsequent dataset and comparison contract

Material follows the user's revised Phase 1 definition: fixed extracted R/M,
uniform albedo0.5 multiplied by independently varied color, no texture image
nodes, normal/bump or displacement. Vary two geometries, two colors and two
lights:8 RGB training renders per natural Material.

Texture fixes the extracted grayscale patch and uses explicit UV mapping;
vary two geometries, two colors, two lights and two nuisance materials:16 RGB
training renders per natural Texture. This gives72 RGB training renders across
the three sources. Shared renderer functions and the existing training pipeline
will be reused through independent Phase 2 manifests/configs.

Keep current Phase 1 settings: SD1.5 revision451f4fe; seed42; embedding LR1e-5;
original K/V LR1e-5; rank64/alpha64 K/V LR5e-5; batch1, constant LR, weight
decay0, cos_weight0, hflip=false and existing object-mask training. Train twelve
independent1000-update runs and save500/750/1000 snapshots. Keep inference
contexts, seeds42/43, PNDM100, CFG3.5,512px,fp16 and safety checking.

Every later contact sheet must show the actual staged RGB training images for
its source/branch (T16 or M8), the natural source, training manifest, method and
checkpoint step. Base/literal controls have no concept-specific fine-tuning.
Extracted maps and generated inference images must not be labelled as training
RGB images. Neutral branch probes and full T/M combinations require matched
visual review; successful execution alone does not establish disentanglement.
