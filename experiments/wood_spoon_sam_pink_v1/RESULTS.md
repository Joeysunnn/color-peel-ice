# Wood spoon SAM + pink mesh preview, 2026-11-05

The current Scheme A pipeline produced one native TRELLIS asset from the full
wood spoon photo, using a freshly generated SAM mask and LCh hue0 pink
conditioning. Execution completed; geometry quality is pending user review.
No S11 dataset, branch initialization, training or transfer was performed.

## Source and execution

- Source: `experiments/natural_material_threeway_v1/assets/wood_spoon_source.png`,
  RGB 2560x2560, SHA256
  `b17fb667be977d2ed879f3c216011e6dde8324951a6f331136a7c52ab2bc4f9d`.
- Code: `7726b732e1b7269ac7a14f4c725aa4ebcb158bfd`, branch
  `codex/wood-spoon-sam-pink-v1`. Full SHA, clean checkout and source hash checked.
- Checkout:
  `/home/r12user5/Documents/Jiawei/colorpeel-checkouts/wood-spoon-sam-pink-7726b73`.
- Run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_sam_pink_v1/20261105__current_sam_pink__7726b73__42`.
- Native asset: `object_001/trellis/native_asset.pt` under the run above.
  Review renders and camera receipt: `object_001/trellis/review/`.
- GPU 0 was checked before execution. One seed42 asset; no retry, camera
  continuation, old mask/mesh substitution or GLB export.

## Observed results

Qwen discovered `wooden spoon`. Grounding DINO + SAM retained one instance,
with detection score 0.9461827874, native mask bbox [1046,98,1512,2419]
and 549859 foreground pixels. The source and SAM input remained 2560x2560.
RGB and mask were resized together to the full 512 canvas without object
cropping or aspect-ratio stretching; its mask contains 21965 pixels.
Pink conditioning preserves that binary SAM alpha and changes no RGB outside
the mask. It is derived from this original source, not a historical pink asset.

The fitted source-view mask IoU is **0.929108205551453**. Camera refinement
**did not converge**: it exhausted 350 Powell evaluations after 72 coarse
evaluations, for 422 total. This is an alignment measurement, not 3D quality
acceptance. The preview shows the complete bowl, handle and hanging hole;
the reconstructed bowl has visible faceting and the hole shape differs.
Hidden surfaces and all-view geometry have not been accepted or validated.

The local review bundle is
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/wood_spoon_sam_pink_20261105/`.
`index.html` links uncropped evidence. `object_001/preview.png` compares original,
SAM, pink condition, fitted clay, PBR and normal. All six panels share the same
horizontal crop x=185..326 and retain full y=0..512 with uniform display scale.
`figure_manifest.json` records hashes and recomputed mask IoU.

## Default integration and bounded fix

Commit `c7805163e04ed20f58aa07c2a69b21910e514ecb` makes
`pink_hue0_for_trellis/v1` the temporary Subject geometry-conditioning default
in the single-image pipeline. Original photos still supply Qwen, S11 photo
anchors and C/M/T sources. Exact-white RGB is preserved to avoid Lab roundoff;
an all-white foreground is explicitly recorded as unchanged. This is a
conditioning policy, not a claim that pink is optimal for every object.

The first attempt at that commit failed in Qwen discovery before SAM or mesh
generation. Its full2560 eager vision attention requested 19.53GiB while only
14.09GiB was free. Evidence remains in sibling run
`20261105__current_sam_pink__c780516__42`.

The corrected commit caps only Qwen's full-frame observation copy to longest
edge1024. Model, prompt and backend remain unchanged; original image and SAM
resolution remain unchanged. The successful run recorded 1024x1024 observation,
actual processor grid [[1,64,64]] and peak Qwen allocation 18.4837465GiB.
This preprocessing change may affect discovery and initializer predictions
for high-resolution images; it is recorded rather than treated as equivalent
to unrestricted full-resolution Qwen input.

## Validation

- Pink preparation, pipeline routing and Subject/training interface tests:
  33 passed locally and in the first clean server checkout.
- Qwen observation, discovery, intake and initializer regression tests:
  67 passed with 2 Torch-dependent skips locally; all 69 passed in the corrected
  clean server checkout.
- Independent code review, actual fresh SAM/pink/TRELLIS execution, archive SHA
  verification, figure hash/IoU checks and visual inspection completed.
- These checks validate this bounded mesh path. The complete four-branch
  training flow was not rerun in this task. User quality acceptance is pending.
