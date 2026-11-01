# Mailbox source-framing72 result, 2026-11-01

Completed three clay previews, visually inspected left/center/right, then
generated all 72 RGB candidates. No training, new model sampling, geometry
repair, camera fitting, or monitoring. Old datasets and outputs remain intact.

Execution commit: `d5014915c52c57d4c1ca33cb87894206eed6c470`.
Branch: `codex/mailbox-source-framing72-v1`.
Server checkout: `/var/tmp/colorpeel-r12user5/checkouts/mailbox-source-framing72-v1`.
GPU: physical2, Tesla V100-SXM2-32GB. Both stage commands exited0.

New experiment parent:
`/var/tmp/colorpeel-r12user5/runs/mailbox_source_framing72_v1/20261101-214327__threeviews72__d501491__42`.
`result/dataset` contains images, masks, captions/concepts, manifest and pairs.
`result/preview` contains the inspected clay views; `result/review/index.html`
shows the complete72. Raw renders, cutouts, maps and geometry buffers are in
the same result folder. Logs and `dataset_bundle.tar.gz` are in the parent.

Local review page:
`C:\Users\lenovo\.codex\visualizations\2026\10\27\01a17b0c-0732-7dd0-9d71-4e4708a1b1ac\mailbox_source_framing72_20261101\result\review\index.html`.

## Verification

- AST,72 unique records /12 groups /one-token captions, and independent code
  review passed before commit/push. Server pulled the same clean commit.
-72 images = blue/green x M[0,1] x R[.15,.50,.85] x3 views x2 HDRIs.
-12 matched groups and84 single-factor M/R pairs; every intervention pair
  produced a nonzero foreground RGB difference (not a perceptual success score).
- All output buffers finite. RGB/mask/raw/cutout pairing and material maps
  passed the original per-record verifier. Per-view mask/normal buffers match
  the inspected clay and remain fixed across material/color/light variants.
- Native geometry and all frozen asset/source/camera/HDRI hashes unchanged.
- Center clay SHA256 `051746ceb5e17907f605e5148e4fa4c2188c5444783d6b338b802f2c54847c5d`
  matches the previously accepted framing preview. All8 corresponding
  center/studio appearance PNGs also match that preview byte-for-byte.
- Downloaded archive SHA256:
  `f4aad661dd16f064c4eed5909b0a48f102906ebe85721efde031924dad1923b3`.
  All352 artifacts listed in the dataset receipt were hash-verified locally.

## Visual review limits

Left/center/right retain bottom cropping (91/157/161 bottom-border foreground
pixels), without top/side image-border clipping. Visual clay inspection found
no newly exposed large cut surface relative to the accepted center preview;
this is limited visual inspection, not a hidden-surface or identity guarantee.
The original flattened crest and contour differences remain. Source is the
actual pink recolored input. Cropping avoids displaying some uncertain lower
geometry but does not complete it. Training effectiveness remains untested.
Dataset candidates remain marked pending user review, with no automatic
training eligibility or training launch.
