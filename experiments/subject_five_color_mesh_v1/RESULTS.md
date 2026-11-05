# Five-color native mesh comparison completed

2026-11-05. All five requested reconstructions and 20 shared-camera clay renders
completed. No extra seeds, generation retries, training, S11 construction or
production-default changes occurred. User quality acceptance remains pending.

Generation/preparation/comparison code: `6726daba4e663c37c368f960a050e8f6dedc22d2`.
The clean server checkout was created from GitHub and its full revision verified.
17 related tests passed both locally and on the server; independent code review
and final evidence review passed.

| Color | LCh hue | Source-view mask IoU | Camera converged | Total evaluations | Vertices | Faces |
|---|---:|---:|---|---:|---:|---:|
| pink | 0 | 0.7804528027 | yes | 270 | 1134277 | 2249226 |
| green | 120 | 0.7765727628 | yes | 386 | 868909 | 1734724 |
| cyan | 180 | 0.7756972112 | no | 422 | 824195 | 1646854 |
| blue | 240 | 0.7469713536 | yes | 304 | 886044 | 1726532 |
| purple | 300 | 0.7723757143 | no | 422 | 801633 | 1593600 |

The unchanged source-camera procedure uses 72 coarse evaluations followed by a
Powell call capped at 350 evaluations / 6 iterations. Totals include the coarse
grid. Cyan and purple reached the evaluation limit; no continuation was run.
Their IoU values are the saved best observed fits, not converged optimum claims.

Pink has the highest recorded source-view IoU, only 0.00388 above green. The
five-color experiment does not establish a clearly superior mesh. All five IoUs
remain below the existing 0.80 Subject guard, and that guard itself is not user
geometry acceptance. No eligible Subject dataset or checkpoint was produced.

## Controls and visual observations

- Original source/image512 SHA256:
  `644a77fe664bc2d9b0c94ada86259ce5a810a530580e754f1b23a40a82906be5`.
- Current binary SAM alpha is identical across all five conditions: 42,397
  foreground pixels, mask SHA256
  `44c8a935359f52c97f3cc788168d0103fc8482295be1f9531efff537a2052d0c`.
- Actual TRELLIS input crops are all 294x294. Model/encoder hashes, precision,
  latent pipeline, sampler configuration, source photograph and seed42 match.
- Each condition is recolored independently from the original orange photograph.
  The historical LCh method preserves L and reduces C only for gamut, with soft
  RGB boundary blending. No historical manual mask repair is applied. The binary
  segmentation alpha, original-photo paths and C/M/T sources stay unchanged.
- The new pink source cutout, actual condition, serialized native asset, fitted
  clay and PBR PNGs match the previous current-SAM pink run byte-for-byte.

The source-fit previews show broadly similar low/wide crest silhouettes and
body proportions, with different edge and end-cap details. None clearly restores
the source photograph's projected proportions. Common-camera views also differ
in end caps, lower contours and crest detail: differences therefore exist in
native geometry, not only in independently fitted display cameras. These are
visual observations, not a quantitative decomposition of shape and camera error.

The four diagnostic cameras are numerically identical across all meshes:
yaw 0/+90/180/-90 degrees, pitch 0.4 radians, roll 0, radius 2.8, FOV 40 degrees,
principal point (0.5,0.5). Native vertices/faces and their coordinate axes are
retained without centering, alignment, scaling or simplification. Opaque clay
uses BC=.5, M=0, R=.5, studio lighting and seed42 for rendering. All 20 masks are
nonempty and do not touch the frame; geometry hashes were unchanged after rendering.

Source-fit clay uses the existing native-alpha/forest review; it is separately
labeled from opaque studio clay. IoU depends on a mask-plus-weak-RGB camera
objective. Gamut chroma reduction can differ by hue. One photo and one seed per
color do not establish a universally best conditioning color.

## Retained results

Server root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/subject_five_color_mesh_v1/20261105__sam_five_hues__6726dab__42`

Each `{pink,green,cyan,blue,purple}` directory contains `inputs/source_receipt.json`,
the recolored cutout, `trellis/native_asset.pt`, `trellis/review/`, generation
receipts and `trellis.log`. GLB export was skipped; native meshes are retained.
`fixed_views/` contains all 20 fixed clay images, masks and comparison provenance.
`pink/pink_previous_reproduction.json` records exact previous-run reproduction.

Local review root:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/five_color_mesh_20261105/`

- `overview.png`: original reference, five inputs, source-fit clay and PBR; the
  same source-canvas crop (0,240,512,512) and uniform layout resizing for every panel.
- `fixed_camera_comparison.png`: four rows of full-frame shared-camera views.
- `fixed_camera_detail.png`: identical (96,128,416,448) pixel crop for all 20 views,
  displayed at 1:1 pixels; no per-mesh zoom and no clipped foreground.
- `index.html`: gallery with links to the original uncropped images.
- `RESULT.json`, `figure_manifest.json`, `repro_outputs/` and `evidence/`: metrics,
  commands, source hashes, reports and downloaded logs. Final independent audit
  verified all 40 fixed image/mask hashes and the five run/convergence records.

Both GPU jobs ended and GPU occupancy was checked afterward. Existing source
assets/checkpoints and other tasks were preserved; no model download occurred.

## Subsequent user review, 2026-11-05

The user judges the Mailbox original-color mesh poor and the pink result better,
while preferring original over pink for the wood spoon. See [the cross-subject
user record](../trellis_conditioning_observations.md). This review is retained
separately from the measured IoUs and unchanged Subject guard above.
