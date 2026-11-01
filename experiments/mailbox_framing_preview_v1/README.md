# Mailbox framing preview

Reuse the frozen native pink-Mailbox asset and dataset renderer. No model
sampling, geometry editing, camera refit, training, or dataset expansion.
The pink source is the recolored foreground original actually used by TRELLIS;
it is not labeled as the orange photograph or a generated training render.

Compare only the source-near orientation: existing centered/full-asset camera
versus the saved estimated source-canvas camera, including its bottom crop.
Both use the same opaque uniform material, full native geometry, studio HDRI,
512 resolution, one depth layer, SSAA1 and reset seed42 for SSAO.
Exterior display pixels remain RGB127 as in the existing dataset; raw images
and masks are retained. No source foreground pixels are pasted onto renders.

Run from a clean committed checkout after local commit/push and server pull:

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export CUDA_VISIBLE_DEVICES=2
PY=/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python
$PY -B -u -m experiments.mailbox_framing_preview_v1.run --output "$NEW_RUN" --stage clay
# Inspect clay_comparison.png and clay/*_mask_overlay.png first.
$PY -B -u -m experiments.mailbox_framing_preview_v1.run --output "$NEW_RUN" --stage appearance
```

Clay is BC=.5, M=0, R=.5. Appearance is blue/green from the frozen dataset
protocol, M=0/1 and R=.15/.85: eight pairs, sixteen renders. Both stages share
one new experiment folder; a completed/failed stage cannot be overwritten.
Appearance checks the completed clay stage's artifact hashes and exact
geometry buffers. Old inputs are hash-checked before and after execution.

Three-column sheets show actual source / centered render / estimated source
framing. Mask overlay legend: white=overlap, orange=source only, cyan=render
only, gray=neither. Centered-source IoU is a framing diagnostic, not a geometry
quality comparison. The saved source-camera fit is approximate; hiding a cut
edge is not geometry completion, and the source crest mismatch may remain.
