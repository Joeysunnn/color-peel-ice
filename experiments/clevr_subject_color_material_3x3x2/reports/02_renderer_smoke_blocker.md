# Renderer v3 paired smoke: file-hash gate blocked

Server commit: `16ccdbeabc85afa55dfee0722a99eb36bf03c799`

The real Blender 4.2.11 CUDA/V100 smoke rendered the paired
`cube-red-metal` and `cube-red-rubber` requests at seed 420000. Camera and
light metadata were identical, both masks were binary and their decoded pixels
were identical, and the material RGB images differed as intended.

The planned byte-level SHA comparison against the accepted v2 metal image did
not pass. A control rerun using the unchanged v2 profile, v2 requests, same
Blender, GPU and seed also did not reproduce the accepted JPEG SHA:

| Comparison | Mean absolute RGB difference | Maximum difference | Changed channel values |
|---|---:|---:|---:|
| accepted v2 vs fresh v2 control | 0.0001373291 | 1 | 108 |
| accepted v2 vs v3 metal | 0.0000381470 | 1 | 30 |

Object/background mask decoded pixels were exact in both comparisons, while
their PNG file bytes differed. This demonstrates that the current byte-hash
gate is stricter than repeatability of the locked renderer itself. The 360-view
render was not started, and the validator was not silently weakened. Continuing
requires an explicit protocol decision to replace byte equality with a locked
pixel-level equivalence criterion while retaining original and rerun hashes.

## Resolution approved 2026-08-26

The user approved `decoded_pixel_equivalence_v1`. Decoded RGB must have maximum
absolute channel difference at most 1 and mean absolute channel difference at
most 0.001. Decoded object/background masks must be exactly equal. Raw artifact
SHA-256 values remain recorded for provenance but are not equality gates. This
changes only realization acceptance; renderer v3, its requests, seeds and all
scientific render parameters remain unchanged.

Server evidence root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260825-093000__clevr_subject_color_material_3x3x2__multiview-render-v3-material__16ccdbe__420000`
