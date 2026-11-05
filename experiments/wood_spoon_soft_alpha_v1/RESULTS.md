# Soft-alpha integration removes the pronounced wood-spoon bowl facets

2026-11-05. One fresh reconstruction with **SAM instance identity + pinned
BiRefNet soft alpha + pink hue0** removes the broad bowl facets visible in the
previous hard-SAM-alpha pink result. Both saved clay and geometric-normal
previews show a much smoother transition. Some fine grid/edge detail remains;
this is a visual finding for this photograph at seed42, not user acceptance or
a guarantee for other photographs, views, colors or seeds.

## Verified input and implementation

Producer commit: `d5b9d15f845e13b2ef26e09e01481b46f5a52559`.

- The current Scheme A photograph and SAM instance were frozen from the
  `7726b73` run. BiRefNet was freshly evaluated on the native original RGB.
- The new **uncolored RGBA512 is pixel-identical to the historical successful
  cutout**: zero changed pixels, maximum channel difference zero. This includes
  alpha. The historical file was used only for this comparison.
- Native soft alpha has one component, uniquely associated with `object_001`,
  with SAM coverage 1.0. Complete opacity values and halo pixels were retained;
  no SAM-boundary clipping, opacity normalization or morphology was performed.
- The semantic SAM mask, original photo, their native counterparts and all
  canonical receipt hashes remain unchanged through matting and recoloring.
  Pink cutout alpha is byte-identical to the separate conditioning alpha.
- The production Subject preparation now invokes soft-alpha preparation before
  recoloring. C/M/T and photograph anchors retain the original SAM/photo inputs.
  Multiple-instance associations are checked against every retained SAM mask;
  a shared component or insufficient coverage fails visibly.

The pinned models, generation settings, seed42, rendering implementation and
temporary pink hue0 default were retained. No geometry/normal smoothing,
decimation, old-mesh reuse or shader change was applied. Joint full-native RGBA
LANCZOS resizing is part of the restored input path, so the before/after
production comparison is an integration verification. The earlier
[isolated alpha experiment](../wood_spoon_input_cause_v1/RESULTS.md) supplies
the causal evidence that hardening alone can trigger this symptom.

## Mesh and framing evidence

| Case | Vertices / faces | Review silhouette IoU |
| --- | --- | --- |
| Previous SAM hard alpha + pink | 409546 / 821522 | 0.9291082056, SAM reference |
| New soft alpha + pink | 416923 / 836782 | 0.9354616938, soft-alpha reference |
| Historical soft alpha + original color | 408127 / 818898 | 0.9478146853, soft-alpha reference |

Against the unchanged **SAM** mask, the new rendered silhouette IoU is
**0.9375624159**. These alignment scores are not measures of surface smoothness.
The new camera fit exhausted 422 evaluations without declaring convergence;
the previews use each asset's saved fitted camera, so the comparison discloses
that their cameras differ. Hidden surfaces and downstream learning were not
validated by this run.

Native mesh SHA256:
`37bab0f3dcd3b80a434e25ce8fddc4cd781f1fa5d104027a18ed140ec92d4e46`.

## Validation, scope and retained results

Local relevant suite: 48 tests, 42 passed and 6 skipped because SciPy is absent.
The clean GitHub server checkout ran all 48 successfully with no skips. Tests
cover component ownership/ambiguity, opacity preservation, unchanged semantic
inputs, source tampering, alpha receipt validation, pipeline wiring and the
actual-render SAM guard. Independent code and evidence reviews were performed.

The real run intentionally reused frozen SAM segmentation, then freshly ran
input preparation, BiRefNet, pink preparation and exactly one TRELLIS generation
under the same full Git revision. It did not rerun discovery/initialization,
expand S11, train any branch, search colors/seeds or resume paused experiments.
The new code path is integrated; a complete new four-branch run was not claimed.

Server run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_soft_alpha_v1/20261105__sam_soft_pink__d5b9d15__42/`.

- `trellis/native_asset.pt`: fresh native mesh and PBR attributes.
- `trellis/review/`: full clay, geometric-normal, PBR and camera evidence.
- `trellis_alpha/`, `trellis_input/`: fresh alpha and pink input receipts.
- `input_comparison.json`, `verification.json`, `status.json`: exact-input,
  semantic-mask, Git and completion checks; quality remains pending user review.

Checkout retained clean at
`/home/r12user5/Documents/Jiawei/colorpeel-checkouts/20261105__soft_alpha__d5b9d15`.
Server test log is the run's parent `tests_d5b9d15.log`.

Local image comparison and downloaded evidence:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/wood_spoon_soft_alpha_20261105/index.html`.
The reviewed archive `soft_alpha_evidence_d5b9d15_v2.tar.gz` has SHA256
`c5c81b2a1664a904488158d38aa6694694c4bb494992e26b40b62523d0f6ff4e`.
An initial evidence-packaging command used system Python without NumPy and
created an incomplete first archive; it is retained separately. Repackaging
with the existing `ice` environment succeeded; generation itself did not fail.
