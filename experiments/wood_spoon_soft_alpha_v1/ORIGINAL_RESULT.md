# New soft alpha + original RGB exactly reproduces the historical spoon mesh

2026-11-05. The requested original-color control completed one fresh TRELLIS
reconstruction at seed42. It uses the new BiRefNet alpha from the preceding
integration run, before pink recoloring. Its generated native output matches
the historical wood-spoon result exactly; no old mesh was substituted.

## Controlled input and exact comparison

The run reused the locked `trellis_alpha/source_receipt.json` produced by
`d5b9d15f845e13b2ef26e09e01481b46f5a52559`, and executed from that same clean
checkout. No executable code change or additional segmentation was needed.
GPU availability and process ownership were checked before running.

Relative to the previous soft-alpha pink run:

- Alpha is byte-identical; original photo and semantic SAM sources are unchanged.
- Conditioning RGB differs at 23326 pixels. Both conditions preprocess to
  464x464; all audited model, sampler, seed, precision, sparse backend, encoder
  and normalization settings match.
- The new input is the unrecolored RGBA cutout from the fresh BiRefNet stage,
  not the historical cutout or an independently resized RGB image.

`experiments.subject_pink_input_repro_v1.compare` confirms exact reproduction
of the historical source RGBA, actual conditioning pixels, configurations,
and all five native arrays: `vertices`, `faces`, `coords`, `attrs`, `origin`.
Shape, dtype, ordering and values match without tolerance, alignment or
resampling; voxel metadata also matches. Even the native file SHA256 is equal:

`a3d581d2bcf28f0ad7ef05ac15a74d24a22decac2c05b5d63fb3a06b3b856aa9`.

The comparison against pink correctly reports different native outputs:
this is the intended color intervention, not a generation failure.

## Visible result and limits

| Same new soft alpha | Original RGB | Pink RGB |
| --- | ---: | ---: |
| Vertices | 408127 | 416923 |
| Faces | 818898 | 836782 |
| Silhouette IoU against soft alpha | 0.9478146853 | 0.9354616938 |
| Silhouette IoU against unchanged SAM | 0.9468160169 | 0.9375624159 |

Clay and geometric-normal previews show smooth bowl transitions without the
previous broad hard-alpha facets. Fine grid/edge detail remains in both color
conditions. For this photograph and seed, pink is not required to reproduce
the historical mesh. These observations do not establish a universally best
color or certify hidden geometry. IoU measures framing, not surface quality.
Both camera fits exhausted 422 evaluations without convergence, and the
original/pink comparison uses their different saved fitted cameras.

Only one new mesh was sampled. No geometry/normal smoothing, S11 expansion,
training, additional seed/color search or change to the production pink
default occurred. User acceptance remains pending. Existing code was reused;
the prior 48-test integration suite was not unnecessarily rerun. Validation
here is actual completed inference plus strict input/config/native comparison
and independent evidence/visual review.

## Retained artifacts

Server run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_soft_alpha_v1/20261105__sam_soft_original__d5b9d15__42/`.

- `trellis/native_asset.pt`: newly generated native mesh.
- `trellis/review/`: complete PBR, clay, normal and camera previews.
- `native_vs_historical.json`, `native_vs_pink.json`: strict comparisons.
- `verification.json`: completion, unchanged alpha, framing and source checks.
- `trellis/repro_outputs/`: command, full Git, model receipts and execution status.

Local comparison and full mesh review:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/wood_spoon_soft_original_20261105/index.html`.
Downloaded archive SHA256:
`a33eeaad0793f5148e805b4eb5075ec696e813721fed96f929d06c9f8ea6c0f9`.
