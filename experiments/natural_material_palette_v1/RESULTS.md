# Material Palette style extraction: mailbox and spoon pilot

Independent server run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_palette_v1/run_20261003_mailbox_spoon_v1/`.
This run trained only the temporary Material Palette LoRA concepts. It did not
train the project's `<M*>` token, alter subject/color experiments, or produce
the full counterfactual dataset.

## Implementation and provenance

- Official [Material Palette](https://github.com/astra-vision/MaterialPalette)
  checkout: `0ddd360467cce8879c9772fb791924b0dc0b9d9b`. Its released
  decomposer `model.ckpt` SHA-256 is
  `03ded80b8e78e365575b6c8e127cadb38f62461f9aab564a68973899d3ab89aa6`.
  Full component hashes are in `deps/assets_manifest.json`.
- The released code's original `runwayml/stable-diffusion-v1-5` reference is
  unavailable. The local Diffusers snapshot uses
  `stable-diffusion-v1-5/stable-diffusion-v1-5`, pinned to revision
  `451f4fe16113bff5a5d22669ed5ad43b0592e9a14`; its text encoder, UNet
  and VAE file hashes are recorded in `deps/assets_manifest.json`.
- Isolated Conda environment uses PyTorch 1.13.1, Diffusers 0.19.3, PEFT 0.5.0
  and PyTorch Lightning 1.8.3. Its `deps/pip_freeze.txt` and setup logs record
  the required compatibility repairs: pip 24.0 for old Lightning metadata,
  MKL below 2024.1 for PyTorch 1.13, plus pinned older Hugging Face packages.
- Source image and mask copies, official crop code hash, crop hashes and Git
  commit are in `input_manifest.json`. Both regions used the official 99% pure
  crop threshold. The mailbox subject-eroded mask required explicit 64px
  patches: the default crop order would stop at one 128px patch and skip it.
  Spoon retained the default scale order and used 128px patches.
- Official temporary concept inversion: 800 steps, prompt
  `an object with azertyuiop texture`, seed 42, trainable UNet and text-encoder
  LoRA. Canonical texture generation: prompt
  `top view realistic texture of azertyuiop`, 1024px, 50 inference steps,
  seed 42. The released code emits one texture per region and omits the
  [paper's](https://openaccess.thecvf.com/content/CVPR2024/html/Lopes_Material_Palette_Extraction_of_Materials_from_a_Single_Image_CVPR_2024_paper.html)
  multi-generation LPIPS ranking. Both raw and source-color-renormalized
  textures were decomposed into Albedo, Roughness and Normal maps.
- The official network saves 8-bit PNG maps, not float tensors. The reported
  medians and IQRs are computed from these PNGs. `canonical/raw/material.json`
  and `canonical/renorm/material.json` under each sample retain map paths,
  hashes and full statistics. Chromatic Albedo is excluded from the material
  concept; grayscale Albedo is archived for audit.

## Outputs

| Sample | Crops | Texture variant | Roughness median | Roughness IQR | Roughness 5–95% | Normal tilt median / 95% |
|---|---:|---|---:|---:|---|---|
| Mailbox painted finish | 100 × 64px | raw | 0.529 | 0.047 | 0.482–0.655 | 6.6° / 23.6° |
| Mailbox painted finish | 100 × 64px | source-color renorm | 0.522 | 0.039 | 0.463–0.604 | 7.5° / 25.9° |
| Spoon bowl | 16 × 128px | raw | 0.596 | 0.031 | 0.549–0.624 | 3.7° / 12.2° |
| Spoon bowl | 16 × 128px | source-color renorm | 0.200 | 0.376 | 0.055–0.588 | 0.3° / 1.9° |

Material Palette **does not predict Metallic**. Each canonical JSON stores
`metallic: null`. The Cycles review uses `metallic=0` as a dielectric renderer
assumption, not an estimate from the natural image. The review uses neutral
gray Base Color and the spatial roughness and normal maps. Spherical UVs were
generated on the existing CLEVR sphere asset, which had no UV coordinates.
There are three fixed lighting renders for each of the four map variants,
12 small preview images total; metadata and image hashes are in each
`render_{raw,renorm}_uv_v2/manifest.json`.

## Visual assessment and decision

- The generated mailbox texture has pronounced red diagonal striations. Its
  predicted Albedo contains local color artifacts, and its Normal map carries
  the same grooves into the neutral-gray sphere render. It does not resemble
  a stable smooth painted finish.
- The spoon's raw texture retains broad bright/dark patches. Its raw roughness
  is high and the dielectric sphere looks matte. Source-color renormalization
  changes the spoon roughness median by −0.396 and raises its IQR from 0.031
  to 0.376; the resulting roughness map contains large dark regions and the
  preview looks much glossier. This is strong sensitivity to the color
  normalization choice, not evidence that either variant is physically right.
- All four A/R/N decompositions and all 12 Cycles previews completed.
  The two source materials remain ambiguous for this single-sample pipeline:
  no metallicity is inferred, and the visible texture/lighting artifacts
  undermine reuse as a canonical material. Do **not** release a full
  counterfactual dataset or use these maps to train `<M*>` from this run.

Earlier SuperMat selected mailbox R=0.3529/M=0.0392 and spoon
R=0.2471/M=0.8471; the IID/VLM pilot selected mailbox R=0.3108/M=0.1173
and spoon R=0.3575/M=0.5970. These estimators have different output spaces
and calibration, so disagreement alone is not a physical accuracy measure.

Review artifacts: `review_sheets/mailbox.png` and
`review_sheets/spoon_positive_control.png` show the source ROI, raw and renorm
textures, A/R/N maps, and three-light renders. The first inversion wrapper
finished all 800 steps but failed while writing its manifest because of a
relative path; the retained checkpoints were subsequently recorded without
retraining. The first decomposition wrapper likewise produced both A/R/N
triplets but expected one; both sets were recorded without overwriting them.
The first spatial renderer attempt found the CLEVR sphere lacked UVs and was
left as `render_raw/` with its failure log. Successful renders use new
`render_*_uv_v2/` directories. All these logs remain in the run root.
