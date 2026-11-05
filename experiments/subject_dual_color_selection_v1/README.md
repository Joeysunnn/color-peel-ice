# Two conditioning colors, one soft alpha, one selected Subject mesh

User request, 2026-11-05. For every object, SAM supplies semantic identity and
the unchanged training/ROI mask. Pinned BiRefNet supplies separate full-native
soft alpha, associated to that instance and jointly resized as RGBA. Preparation
then generates exactly two seed42 candidates from the same alpha: original RGB
and the existing pink hue0 recolor. No color/seed search or automatic retry.

Each candidate is retained in `trellis_candidates/{original,pink}`. A generation
failure is recorded independently and does not prevent trying the sibling.
The selected receipt, native mesh and saved camera are passed directly to the
existing Subject11 renderer. Original photo anchors and C/M/T inputs are unchanged.

## Selection protocol

1. Verify common source/photo/SAM, soft alpha, model/encoder, sampler, seed,
   normalization and generation Git. A successful run must contain one mesh.
2. Reject unusable geometry: malformed/nonfinite arrays, invalid face indices,
   no positive triangle area or empty geometry. Record degeneracy; do not require
   watertight/manifold topology or rank by face count.
3. Keep each candidate's saved source camera and existing review guard. Render
   an opaque geometry silhouette, without learned transparency, at512 resolution.
   Apply the unchanged SAM IoU>=0.80 and frame-clipping guard.
4. Measure `IoU` and symmetric mean boundary distance. Boundaries use8-neighbor
   erosion with zero padding, including holes and disconnected pieces. Each
   direction has weight0.5; divide pixel distance by the SAM bounding-box diagonal.
5. If both pass and `abs(delta_IoU)<=0.01` and `abs(delta_distance)<=0.005`, choose
   randomly using a recorded seed derived from the declared seed, photograph,
   native SAM hash and object ID. Otherwise maximize `IoU - normalized_distance`.
   If only one passes, choose it. If neither passes, block Subject preparation.

Thresholds and score are fixed engineering heuristics, not calibrated perceptual
quality measures. They choose a mesh plus saved-camera combination by source-view
alignment; they cannot judge bowl facets, actual depth or unseen surfaces. Saved
cameras retain the same fit budget and weak-RGB fitting differences. Selecting
a candidate does not constitute user quality acceptance.

`trellis_selection/selection.json` records candidates, failures, scores, source
locks, thresholds, random seed/draw and the decision. Neither candidate is deleted
or overwritten. A blocked Subject does not automatically block eligible C/M/T.

## Bounded validation without training

The single-image pipeline uses `prepare_subject_geometry`. This entry exercises
that same production helper for one frozen retained SAM object, ending after
selection without Qwen reinitialization, S11 expansion or branch training:

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT="$COLORPEEL_RUN_ROOT"
/home/r12user5/miniforge3/envs/ice/bin/python -B \
  -m experiments.subject_dual_color_selection_v1.run \
  --objects-json /absolute/path/segmentation/objects.json --object-id object_003 \
  --output "$COLORPEEL_RUN_ROOT/subject_dual_color_selection_v1/NEW_RUN" --gpu 0
```

Use a tested, pushed clean checkout with full SHA verification and check GPU
occupancy/ownership first. Model files stay pinned and offline. The real new
input preparation and both meshes must share that checkout revision.

Completed production-helper verification: [RESULTS.md](RESULTS.md).
