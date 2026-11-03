# Orange Mailbox: TRELLIS.2 versus SuperMat, extraction and rendering only

Use the same original orange RGB and full-object binary mask for both models.
`prepare` preserves foreground pixels and zeros hidden RGB in one shared RGBA
file. SuperMat uses its official gray-composite original512 preprocessing;
TRELLIS.2 uses its official foreground crop/resize and black composite. These
are different model preprocessors, not identical conditioning tensors.

TRELLIS uses the existing pinned official1024 cascade, seed42, verified public
DINOv3 conversion and V100 FP16/xformers/explicit_gemm adaptations. Always pass
the new source receipt, pinned model revisions and `--skip-glb`; the old default
pink input must not be used. SuperMat runs again from the same RGBA, with its
existing checkpoint and512px, seed42, one-step official estimator. No model
weights, trainer, initialization policy or running job are changed.

`export_raw` retains unquantized visible TRELLIS A/R/M at the estimated source
camera. Base Color is the native display-domain attribute, not the shaded PBR
render; do not apply an additional gamma. M/R are linear scalar parameters.
Keep raw out-of-range diagnostics; clip to[0,1] only for scalar rendering.
SuperMat's official separate8bit M/R outputs retain their existing decoding.

`compare_maps` uses the historical small-painted-lid ROI. Intersect it with
source foreground, TRELLIS geometry mask>=0.9 and predicted alpha>=0.9, erode2px and use
exactly the same surviving image coordinates for both medians. Require at
least50pixels and50% of the initial ROI; otherwise save diagnostic figures and
fail without publishing scalar materials. This is approximate image-plane
overlap, not recovered surface correspondence. Inspect the overlay before
interpreting or rendering the extracted values. No reconstruction error score
against the source RGB is claimed: photographs are not ground-truth albedo.

`render_compare` feeds the two scalar materials into the existing Material8
renderer: sphere/cube x red/blue x soft_front/side_directional,512px,64samples,
Standard/sRGB, matched per-pair seeds, camera, geometry, lights and uniform
Base Color=0.5*palette.16 images total. This isolates M/R, not spatial A/R/M
texture transfer. Four-thread CPU rendering is available; `--gpu N` uses the
existing V100 path after checking occupancy. Images and manifests stay outside
the source checkout. There is no training or composition stage.

Entry points, run from a clean pushed checkout (all output directories fresh):

```bash
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export PYTHONDONTWRITEBYTECODE=1
python -B -m experiments.trellis_supermat_compare_v1.prepare --output "$RUN/input"
python -B -m experiments.subject_trellis_counterfactual_v1.run \
  --output "$RUN/trellis" --encoder "$TRELLIS_ASSETS/dino_timm_v2/model" \
  --source-receipt "$RUN/input/source_receipt.json" \
  --model-revisions "$RUN/input/model_revisions.json" --skip-glb
python -B -m experiments.natural_material_supermat_v1.run_supermat_batch \
  --python "$SUPER_PY" --supermat-root "$SUPER_ROOT" --checkpoint "$SUPER_CKPT" \
  --base-model "$SUPER_BASE" --input-dir "$RUN/input/inputs" --output-dir "$RUN/supermat" --device cuda:0 --seed 42
python -B -m experiments.trellis_supermat_compare_v1.export_raw --run "$RUN/trellis"
python -B -m experiments.trellis_supermat_compare_v1.compare_maps \
  --source "$RUN/input/inputs/mailbox.png" --roi "$RUN/input/reviewed_roi.png" \
  --supermat "$RUN/supermat/mailbox" --trellis-review "$RUN/trellis/review" --output "$RUN/maps"
blender --background --python-exit-code 1 --python experiments/trellis_supermat_compare_v1/render_compare.py -- \
  --supermat "$RUN/maps/supermat/material.json" --trellis "$RUN/maps/trellis/material.json" --output "$RUN/renders"
python -B -m experiments.trellis_supermat_compare_v1.render_compare --grid "$RUN/renders/manifest.json"
```

Use the matching existing TRELLIS/SuperMat/Pillow Python environments for each
command. Preserve exact executed commands, versions, input/model/code hashes,
logs, masks and full-size outputs in the external run. Completion means an
auditable comparison is available; neither method becomes physical truth or a
new training default based on this single example.

Paired masks are checked by exact decoded RGBA8 pixel hashes, while original
PNG file hashes remain provenance. Blender embeds different timestamps and
render durations even when every mask pixel is identical. The first attempt
stopped on that metadata-only difference and is preserved; use a fresh render
directory for the corrected check. `report --render-dir` explicitly selects
that completed attempt without altering earlier files.
