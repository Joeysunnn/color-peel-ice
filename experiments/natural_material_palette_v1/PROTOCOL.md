# Material Palette style material extraction

This is an independent natural-image material baseline using the official
[Material Palette](https://github.com/astra-vision/MaterialPalette) pipeline:
material-region crops → temporary SD 1.5 LoRA concept → canonical top-view
texture → pretrained decomposition into Albedo, Roughness and Normal maps.
Use the user's mailbox painted coating and a separate spoon positive control.
Do not modify subject/color checkpoints, generate the 96-image counterfactual
dataset, or train the project's `<M*>` token.

The [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Lopes_Material_Palette_Extraction_of_Materials_from_a_Single_Image_CVPR_2024_paper.html)
and released code have an important difference: the paper ranks multiple
generated textures using LPIPS, while the released pipeline produces one
generation per region without that ranking. This pilot follows the released
implementation. Record the generated texture and flag geometry, perspective,
lighting, color or seam artifacts before treating its decomposition as useful.

The official decomposer has Albedo, Roughness and Normal heads, **not a
Metallic head**. Preserve its full maps. Remove chromatic Albedo from the
canonical concept; retain a grayscale Albedo audit image. The initial renderer
uses neutral gray Base Color, the spatial Roughness and Normal maps, and
`metallic=0` as an explicit dielectric preview assumption. It must not be
reported as measured metallicity. Keep all output images and exact input,
checkpoint, script, model-revision, seed, crop, rendering and Git hashes.
The released predictor saves 8-bit PNG maps. Report that quantization and do
not claim float precision from them. Keep both raw and source-color-renormalized
texture decompositions; use raw as the primary analysis before excluding
chromatic Albedo, since renorm explicitly matches source color statistics.

The official crop function requires at least two square crops with ≥99%
material coverage. On the mailbox subject-eroded mask, the default search
stops after finding one 128px crop and would skip the material. Explicitly use
64px for mailbox only, with the same 99% threshold. The spoon uses official
scale order. Small mailbox crops risk amplifying shape and illumination cues;
inspect them and the top-view generation. Do not silently broaden either ROI.

Use separate `samples/mailbox` and `samples/spoon_positive_control` directories
under a new server run root. Keep official repository and weights in that
experiment's `deps/`, outside older experiment outputs. The original SD 1.5
repository named in the released code is unavailable; use the pinned
`stable-diffusion-v1-5/stable-diffusion-v1-5` Diffusers mirror and record its
revision and component hashes. Environment repairs stay in an isolated Conda
prefix. Run only small three-light renderer checks after decomposition.
