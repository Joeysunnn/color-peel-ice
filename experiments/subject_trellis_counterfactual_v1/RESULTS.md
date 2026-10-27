# Pink Mailbox: single-asset review, 2026-10-27

## Outcome and scope

TRELLIS.2 generated exactly one asset from the frozen pink foreground original.
Native geometry/PBR attributes, source-canvas review images and a portable GLB
preview were retained. This establishes that the adapted pipeline runs on the
server; subject identity, Albedo and material accuracy remain unaccepted.
No counterfactual dataset, training or monitoring was started.

## Runtime and model access

- Server: research12, Tesla V100-SXM2-32GB; isolated runtime below
  `/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env`.
- Torch2.6/cu124, xformers0.0.29.post3, Transformers4.57.3.
- Official TRELLIS.2 source: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`.
- TRELLIS.2-4B weights: `af44b45f2e35a493886929c6d786e563ec68364d`;
  TRELLIS-image-large decoder: `25e0d31ffbebe4b5a97464dd851910efc3002d96`.
- V100 lacks native BF16. Flow torsos used the official `convert_to(float16)`
  API, retaining the implementation's FP32 boundary layers.
- Default masked-implicit sparse GEMM aborted with `LLVM ERROR: Unsupported
  layout`. The documented `explicit_gemm` implementation passed numeric sparse
  convolution, attention, rasterizer and CuMesh checks, then the complete sample.
  Official sources and weights were not patched. This is adapted inference,
  not an exact default BF16 reproduction.

The gated Facebook DINOv3 endpoint returned403. The public same-model
`timm/vit_large_patch16_dinov3.lvd1689m` distribution was used at revision
`30c1109559f65dea34316b0d4842d35c5771fe11`; DINOv2 was not substituted.
Its tensors were mapped into the original Transformers DINOv3 implementation.
For the actual pink input, normalized TRELLIS conditioning features agreed with
timm at512/1024, with max absolute errors1.4067e-5/4.7207e-5 and unchanged
atol/rtol1e-3. Raw pre-normalization features at1024 did not pass allclose;
that diagnostic and the earlier failed attempt were retained. No byte-level
comparison against the inaccessible Facebook checkpoint is claimed.

## Input and generation

Source run:
`/var/tmp/colorpeel-r12user5/runs/subject_marble_foreground_v1/20261026-131331__foreground30_plus5__7bc1a86__42`.
Input `source/pink/cutout512.png` SHA256:
`e263e3f1dd319426baa6cb8f72ff56495230962eea40659497d1cf49b665c7c1`.
The alpha matches the frozen mask; transparent hidden RGB is zero. The source,
mask, dataset manifest and preparation receipt passed their frozen hash checks
again after the GLB export. No existing source or completed output was edited.

Official alpha-based preprocessing, one1024_cascade sample, seed42; official
12-step samplers and CFG settings were retained. No rembg was needed. Native
sampling took35.864s, excluding startup, downloads, review and export.

Generation commit: `e0ca3732a0bd47974e14b2175bf9a82aa1188f2c`.
Native run directory under `/var/tmp/colorpeel-r12user5/runs/subject_trellis_counterfactual_v1`:

`20261027-110302__pink_asset__e0ca373__42`

- `native_asset.pt`: 919,180 vertices, 1,836,844 faces; finite geometry/attributes.
- Native SHA256: `caebd9cfa959caf64964c765d320039bd0a3d482f5fabe7c19e400d26161285e`.
- `source_cutout512.png`, `actual_condition.png`, `pipeline.json`, first `review/`.
- The secondary million-face-target export stopped before saving GLB. At
  11:27 HKT its process was absent, its log ended at xatlas charts39%, and
  `repro_outputs/status.json` still said `started/glb_export` with one generated
  asset. The cause was not established. That stale receipt and log are retained;
  they must not be read as a successful complete export or as failed sampling.

## Native review

Same-asset review commit: `b4a76cd7f3c736ca869340914bf0e713939dea5a`.
Review directory under the same run root:

`20261027-111312__camera_review__b4a76cd__42`

The original512 canvas is preserved, including its existing bottom crop. A
camera fit includes yaw, pitch, roll, distance and principal point; it changes
neither mesh nor native attributes. Refinement reached its evaluation limit.
Original-canvas mask IoU improved from0.79475 to0.81433; this remains an
estimated source view, not a recovered calibrated camera or an identity score.

`contact_sheet.png` shows source, PBR, clay, Base Color, Metallic, Roughness and
normal. Per-panel PNGs, fitted mask, camera matrices and rendering details are
in `review.json`; `saved_asset_source.json` records hashes and zero new samples.
Lighting is the fixed official forest HDRI. The clay material is a separate
attribute clone. Native PBR output receives only the official display transform;
maps are clipped for display without texture or color replacement.

Observed limitations requiring user inspection:

- Base Color shifts bright pink toward red and retains dark shading patterns.
- The mailbox shape has recognizable relief, but pose, silhouette and details
  differ; the camera fit alone cannot resolve geometry error.
- Native Metallic ranges -0.0014085 to0.0017107; Roughness ranges0.9982951
  to1.0020866. Display clipping produces nearly black M and white R. These are
  generated estimates, not evidence of recovered physical reflectance.
- No claim of Albedo preservation or subject/material disentanglement follows
  from a successful render. This asset has not been accepted into a dataset.

## Portable preview

Export commit: `11eeff8da34f0d16cec672718099c21a5c5fd460`.
Independent export directory under the same run root:

`20261027-113245__preview_glb__11eeff8__42`

`export_saved` loaded the retained native asset without a model call. Official
cleanup/simplification and UV baking used target100,000 faces, texture2048 and
remesh=False. Export took23.310s. `mailbox_preview_lod.glb` has58,661 vertices,
99,917 faces, RGBA Base Color and RGB packed R/M textures, both2048×2048.
GLB SHA256: `7c8a18cefe2d991d625c2738c74247bafe31b705c4e3a15f775aee70d0c4b77d`.

Reloading the GLB with trimesh verified those counts, finite positions/UVs and
both textures. `export_receipt.json` records completed export, zero new model
samples and unchanged native input hash. This is a preview LOD; the displayed
native review uses the full pre-export mesh and must not be described as a
pixel-identical rendering of the simplified GLB.

All source edits were made locally, committed and pushed, then pulled into
clean server checkouts. Native/model/cache artifacts remain outside Git.
