# ICBench selected 36 objects: pink RGB and GT binary alpha

Use the 36 `Pilot Eligible=YES` rows from `Manual Review` in the user's
`ICE_D1_mask_manual_review.xlsx`. These span 29 images. The external selection
JSON records workbook SHA, selection cells, stable sample/mask IDs and archive
member hashes. Bind each `ICE_stage-one-D1-results/{sample}/img.jpg` to
`D1_masks/{sample}/{mask}.png`; use the 512px benchmark image, not `original.jpg`.

For each selected GT object:

1. Preserve the supplied binary mask exactly (0/255), including all its components.
   No SAM, BiRefNet, threshold tuning, mask repair, erosion or dilation.
2. Reuse the established deterministic Lab/LCh hue=0 pink recoloring inside the
   GT support. Its historical RGB edge blending and exact-white protection stay
   unchanged; they do not soften the geometry alpha. Set transparent RGB to zero.
3. Generate one native TRELLIS.2 mesh, seed42, cached pinned models and the existing
   1024 cascade settings. Leave `--fixed-crop` unset: official alpha>204 automatic
   cropping applies independently to every object, including fractional boxes.
4. Render unchanged geometry with the same four views as the previous experiment:
   this mesh's source reference camera, then yaw0/+90/-90 at pitch0.4, radius2.8,
   FOV40. Retain the previous flat-face Lambert clay lighting and normal/mask views.
   No geometry selector or quality-based filtering; results await user review.

The per-object `geometry_views/comparison_clay.png` contains the four clay views.
`index.html` presents all 36 entries, including explicit failures. Native assets
are `image_NNN__mask_ID/trellis/native_asset.pt`. Failure of one object does not
stop the other objects; no retries or old mesh fallback. No branch training.

## Launch

Local tests/review and fork push first; use a new clean server checkout at the
exact full commit. Set both `COLORPEEL_RUN_ROOT` and `COLORPEEL_OUTPUT_ROOT` to
`/home/r12user5/Documents/Jiawei/colorpeel-runs`. Validate server input hashes
against the local archives and check GPU occupancy/process ownership first.

```sh
python -B -m experiments.icbench_gt_mesh_v1.run \
  --selection EXTERNAL_SELECTION_JSON --output FRESH_RUN --gpus 0 1 2 3
```

One serial queue per assigned GPU generates and then renders each object.
The batch writes status and the gallery itself. Only finite startup verification
is requested; do not add a monitoring process or assess generated quality.
