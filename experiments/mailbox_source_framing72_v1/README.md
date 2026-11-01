# Mailbox source-framing72

User accepted the restored-source framing and selected three views / 72 images.
First inspect three clay views; then build blue/green x M[0,1] x R[.15,.50,.85]
x yaw[-.25,0,+.25] x studio/courtyard. No training or new model sampling.

Reuse the full frozen native asset and official PBR renderer from the original
dataset. Center uses saved camera matrices verbatim; side views change yaw
only, keeping radius, pitch, roll, principal point and FOV. No bbox centering,
fit-to-frame, per-material alignment or geometry repair. Require bottom crop
without new top/side image-border clipping. All geometry/material buffers are
verified using the original dataset's checks. Each view's geometry must match
its inspected clay preview. The old source is the pink recolored original,
not the orange photo and not a MARBLE candidate. Exterior RGB127 remains the
existing dataset convention; actual rendered foreground pixels are retained.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp CUDA_VISIBLE_DEVICES=2
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
PY=/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python
$PY -B -u -m experiments.mailbox_source_framing72_v1.run --output "$NEW_ROOT/result" --stage views
# Inspect result/preview/clay_three_views.png before the second command.
$PY -B -u -m experiments.mailbox_source_framing72_v1.run --output "$NEW_ROOT/result" --stage dataset --preview-reviewed
```

Keep logs inside the new experiment parent. Both stages refuse overwrites;
failed attempts remain intact. Source hashes and clean Git states are checked
before/after. Full dataset: RGB images, masks, captions/concepts, raw renders,
cutouts, material maps, 12 matched blocks / 84 one-factor M/R edges, six review
sheets and HTML. No automatic training eligibility or geometry-accuracy claim.
Preserving source crop hides some unsupported geometry but cannot repair the
known flattened crest or establish complete-object identity.
