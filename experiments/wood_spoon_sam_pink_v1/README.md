# Wood spoon through current Scheme A + pink TRELLIS input

User request, 2026-11-05: temporarily use pink geometry conditioning for all
single-image inputs and inspect the existing full wood spoon through that flow.
Production integration is in `single_image_pipeline.py`; this bounded runner
stops after mesh/review generation and does not initialize/train branches or
construct S11/C/M/T datasets.

The frozen original is
`experiments/natural_material_threeway_v1/assets/wood_spoon_source.png`,
SHA256 `b17fb667be977d2ed879f3c216011e6dde8324951a6f331136a7c52ab2bc4f9d`,
RGB 2560x2560, containing the bowl, full handle and hanging hole.

1. Qwen whole-object discovery, Grounding DINO boxes and SAM masks run fresh
   through the current Scheme A workers, with the existing duplicate policy.
   Qwen alone observes a full-frame copy capped at longest edge1024; its actual
   processor grid is recorded. Original source and SAM input remain2560x2560.
2. Original RGB and binary instance mask are mapped together to the 512 canvas,
   without object cropping or aspect-ratio stretching.
3. LCh hue0 recoloring with historical gamut/soft-RGB rules changes only the
   TRELLIS condition. Binary SAM alpha and original-photo paths stay unchanged.
   Exact-white foreground RGB is preserved to avoid Lab roundoff above L=100;
   an all-white foreground is explicitly recorded as unchanged, not a pink edit.
4. One seed42 native TRELLIS mesh per retained instance, existing pinned models,
   converted encoder, sampler settings and camera-fit budget. No retries.

No older spoon BiRefNet alpha, bowl-only mask, old mesh or Mailbox asset is used.
GLB export is skipped; complete native mesh/PBR attributes and fitted source-view
PBR/clay/normal/mask images are retained. Successful execution is not mask or
geometry quality acceptance. No automatic camera continuation is included.

From a reviewed clean GitHub checkout, with both output-root variables set:

```bash
/home/r12user5/miniforge3/envs/ice/bin/python -u -B \
  -m experiments.wood_spoon_sam_pink_v1.run \
  --output /home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_sam_pink_v1/NEW_RUN \
  --gpu 0
```

Each subprocess checks GPU occupancy and uses offline cached assets. Outputs are
fresh and outside Git; raw discovery, segmentation scores/overlays, source and
pink receipts, commands and failed evidence are retained.

The first attempt at `c780516` failed during Qwen discovery: eager vision
attention on the full2560 image requested19.53GiB with14.09GiB free. It produced
no SAM masks or native meshes. That run is retained. The observation-size fix
is a preprocessing change for Qwen, not a new mask rule or TRELLIS color search;
one corrected run uses a new output and fresh checkout.
