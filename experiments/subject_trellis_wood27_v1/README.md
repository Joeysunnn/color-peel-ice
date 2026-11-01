# Wood spoon Subject27 dataset only

The user specified24 synthetic images plus the original photograph and its two
LCh recolors: **27**, despite the earlier label "25-image version". No training,
inference sweep, new geometry, segmentation or monitoring is started here.

- Reuse the exact RGB, binary masks and captions from the frozen Subject72 data.
- Keep blue/green, M={0,1}, R={0.15,0.85}, left/center/right, studio only.
- The complete24-image Cartesian product retains its original source order;
  reject missing/duplicate cells. Do not render again or change old files.
- Append the original full-canvas wood photograph resized2560→512 with LANCZOS,
  then its blue and green LCh recolors. Preserve its original white background,
  framing, wood grain and illumination. The full-resolution original is copied
  into `sources/` for provenance; it is not an extra dataset row.
- Reuse the existing pinned BiRefNet soft alpha, derive the photo object mask
  with alpha>=128, and use the same binary object mask for all three photos.
  No cropping, mask repair, new segmentation or background neutralization.

The inherited `natural_subject_recolor_pilot.recolor` changes only CIELAB D65
hue: blue240°, green120°. It retains per-pixel L and source C, reducing C only
for legal sRGB at fixed L/h (32 bisection iterations, margin0.995). Soft blending
uses the original alpha8×257 /65535 in sRGB. These Lab hue angles follow the old
Subject recolor palette; they are not the Lab hues of the TRELLIS constant RGBs.

Exact RGB white is left untouched as neutral. The existing conversion matrix
maps white to L=100.0000038667, outside the old helper's strict bounds. Only these
white pixels are excluded from the editing mask; the helper and the photo
training masks are unchanged. Pixels outside the editing mask stay byte equal.
This handles numerical white without clipping RGB/Lab or altering geometry.

The synthetic captions are unchanged. Photo captions identify the `<S*>` spoon
and wood surface, with blue/green words on the recolors. Each row has one image,
one binary mask, one caption and one `<S*>`; concepts keep this exact pairing.
Photos retain their actual illumination and are labelled `original_photo`, not
studio renders or known M/R. The27 images are a mixed synthetic/photo dataset,
not27 calibrated physical counterfactuals.

Deploy through local commit/push, then a separate clean server clone/pull:

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.subject_trellis_wood27_v1.prepare --output "$FRESH_DATASET_RUN"
```

The output contains `dataset/{concepts.json,training_assets_manifest.jsonl,assets}`,
source copies, frozen protocol, source hashes, LCh measurements/maps, a27-image
contact sheet and portable HTML review. All new rows remain pending user review
and are not automatically used for training. Source hashes are rechecked after
building. A sibling ZIP packages the review and actual dataset, with its SHA
printed at completion. Failed/final directories are never overwritten.
