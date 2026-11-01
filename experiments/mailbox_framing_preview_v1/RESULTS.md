# Mailbox framing comparison, 2026-11-01

Completed two clay renders, visually inspected them, then completed sixteen
appearance renders (blue/green x M=0/1 x R=.15/.85 x two framings). All use
the same source-near orientation and studio light. No training, new TRELLIS
sampling, geometry repair, or automatic monitoring was started.

Execution commit: `731a78a20b4d520197df321db1ae1202c6f4d073`.
Branch: `codex/mailbox-framing-preview-v1`.
Server checkout: `/var/tmp/colorpeel-r12user5/checkouts/mailbox-framing-preview-v1`.
GPU: physical 2, Tesla V100-SXM2-32GB.

New experiment root:
`/var/tmp/colorpeel-r12user5/runs/mailbox_framing_preview_v1/20261101-212732__clay_then_appearance__731a78a__42`.

All images, cameras, masks, source copies and receipts are in `result/`;
`clay.log`, `appearance.log` and `review_bundle.tar.gz` share the parent.
Run the README commands with `--output <experiment-root>/result`, clay first.
The initial shell redirection failed because the new study parent did not
exist; no Python process or output was created by that command. Creating the
new experiment parent and placing logs inside it resolved the launch issue.
Both actual rendering commands exited 0.

Local review:
`C:\Users\lenovo\.codex\visualizations\2026\10\27\01a17b0c-0732-7dd0-9d71-4e4708a1b1ac\mailbox_framing_preview_20261101\result\index.html`.

## Checks

- Local AST check and independent code review passed before commit/push.
- Server pulled the same clean execution commit before running.
- Native asset, actual pink source, source-camera receipt and studio HDRI
  matched frozen SHA256 hashes before and after both stages.
- Full native vertices/faces remained unchanged. All outputs finite;
  rendered BC/M/R values matched requested controls within atol1e-6.
- Mask and normal buffers were identical across all materials within each
  framing, including agreement with the preceding clay stage.
- New archive SHA256:
  `563b00daac4b62de7f0d9dbd64e7a4a8422adad1e04de751faa4469aa3c47ae6`.
  Download matched it; all 68 artifact hashes listed in the appearance-stage
  receipt were independently checked locally.

## Visual findings and limits

The centered mask bbox is [92,146,447,377], with no bottom-border pixels.
The source-framing bbox is [136,329,440,511], with 157 bottom-border pixels.
Source-framing mask IoU is 0.8143274854, reproducing the saved approximate
camera alignment. These are composition diagnostics, not identity acceptance.

Restoring the saved source framing places the lower region outside the image
again, avoiding full display of that generated lower contour in the center.
It does not reconstruct the missing region. The raised pointed crest in the
source is still flattened in the generated asset, and visible contour/pose
differences persist. The effect is present across the appearance pairs;
specular highlights vary with the fixed M/R controls as expected.

The first column is explicitly the pink recolored foreground original used
for the existing TRELLIS asset, not the orange source photograph. All three
columns retain their full512 canvas; the source is not warped or recolored
again. Gray rendering materials are diagnostic, not photographic ground truth.
Training suitability and improved learned identity remain untested and require
user review. Existing datasets, checkpoints and experiment outputs were not edited.
