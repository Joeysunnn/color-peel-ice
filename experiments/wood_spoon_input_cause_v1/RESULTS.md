# Historical alpha hardening reproduces wood spoon facets

2026-11-05. The controlled test identifies **historical alpha binarization as
a sufficient trigger** for the pronounced bowl facets in this image at seed42.
It does not require SAM, recoloring, a different crop, changed RGB resampling,
different models or a different renderer. This is a case-specific causal
result, not a claim that every binary mask fails or every soft mask repairs it.

## Exact historical replay

Current code `02ac8754f8d6cb76175a85537a2cc5330d3d2dd9` regenerated the historical
wood spoon from the exact historical source receipt.

- Decoded source RGBA and actual 464x464 conditioning pixels match exactly.
- Configuration, model hashes, encoder receipt, samplers, FP16 adaptation,
  explicit-GEMM backend, seed42 and voxel metadata match.
- All five native arrays (`vertices`, `faces`, `coords`, `attrs`, `origin`)
  match exactly in dtype, shape, order and values, without alignment/tolerance.
- Even native asset SHA256 matches:
  `a3d581d2bcf28f0ad7ef05ac15a74d24a22decac2c05b5d63fb3a06b3b856aa9`.
- The regenerated bowl is smooth in saved clay and geometric-normal previews.

Thus current code/runtime can reproduce the old result; observed faceting does
not require an unexplained runtime or source-code drift.

## Single intervention

From that same historical RGBA512, `prepare.py` changes alpha to
`(old_alpha > 204) * 255`, exactly the support used by the official crop test.
It preserves RGB byte-for-byte on all 21951 retained pixels, zeros RGB only in
newly transparent pixels, and preserves the historical alpha as evidence.

The crop remains **[23,19,487,483]** and actual conditioning remains **464x464**.
The model, seed, samplers and generation source are unchanged. The result now
has broad, visible facets in both clay and geometric-normal images.

| Result | Vertices / faces | Source-view silhouette IoU |
|---|---|---|
| Historical replay | 408127 / 818898 | 0.9478146853 |
| Only historical alpha hardened | 378953 / 760826 | 0.9295152238 |

Both camera fits exhausted 422 evaluations. IoU is alignment evidence, not a
measurement of bowl smoothness or geometry approval. Different fitted cameras
are disclosed; the visible face-normal patches are derived from mesh geometry.

## Where this enters the current pipeline

`src/methods/colorpeel_ice/single_image_subject.py` resizes the SAM mask with
NEAREST and copies that binary mask directly into the RGBA cutout alpha. The
historical path instead preserves BiRefNet soft alpha and resizes RGBA jointly.

The [pinned official TRELLIS preprocessing](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/pipelines/trellis2_image_to_3d.py)
uses alpha for the crop, then multiplies RGB by alpha and returns an RGB image.
Consequently alpha is not an independent mask channel passed to the image
encoder: changing alpha changes the composited RGB conditioning pixels.

Historical interior alpha is mostly254, not255. Within the retained support,
hardening changes the composite by a mean1.179/255 per channel. It also removes
2293 nonzero soft-alpha pixels outside the >204 support. This experiment changes
both those effects together; it does **not** prove that interior one-level
brightness changes alone, or boundary removal alone, cause the facets.
The network-internal amplification mechanism was not separately isolated.

The previous SAM/original-color control already ruled out pink as necessary.
This new test additionally shows that SAM boundary differences and the464-to462
crop change are not necessary for facets to appear. Those factors may still
influence geometry. No universal quality claim follows from one image/seed.

## Repair implication and stop condition

The next repair should keep semantic instance masks separate from TRELLIS
appearance alpha. SAM can continue supplying instance discovery/ROI/training
masks, while a separately audited matting/RGBA preparation supplies TRELLIS.
The historical BiRefNet preparation is a reproduced reference for this spoon.
Arbitrary edge blur, constant alpha254, switching color, vertex-normal averaging
or reusing the old mesh have not been established as fixes for the automated
pipeline. No such production change is included in this diagnostic.

Two reconstruction runs completed: exact replay and the single alpha
intervention. Further mask/crop/RGB trials stopped once the isolated operation
reproduced the target symptom. Production pink default remains unchanged;
no training, S11 expansion, geometry smoothing or quality acceptance occurred.

Follow-up: the separately recorded [soft-alpha integration verification](../wood_spoon_soft_alpha_v1/RESULTS.md)
now implements this separation and tests one fresh pink-conditioned spoon mesh.
The diagnostic runs and conclusions above remain the historical record.

## Evidence and validation

Server root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_input_cause_v1/`.

- `20261105__historical_replay__02ac875__42/`: native comparison and TRELLIS output.
- `20261105__old_hard_alpha__87581d3__42/`: input receipt, original alpha,
  native comparison and TRELLIS output. Native asset is `trellis/native_asset.pt`.

Intervention code: `87581d3ad6f10cab52c9cab93e04870191295aa7`.
Both runs used fresh clean GitHub checkouts with full SHA and idle GPU checks.
Two CPU tests passed locally and on the server. Independent code, receipt and
visual audits confirmed exact RGB/crop controls and actual conditioning pixels.
Downloaded evidence archive SHA256:
`35cfcc2e755332ff512120ae6a6321630d717879568bd417818b1642e2d35ad4`.

Local evidence and comparison:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/wood_spoon_input_cause_20261105/index.html`.
