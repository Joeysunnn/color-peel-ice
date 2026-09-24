# Orange color and selected metal composition diagnostic

## Sources and fixed comparison

The project owner selected the original ground-reflection `<M*>` source recorded
in `experiments/material_token_local_pilot_v1/configs/selected_material_source.json`.
The independent orange `<C*>` token-local short run used the nine previously
verified emission images from the natural-image subject/color mainline. Its
server run is
`/home/r12user5/Documents/Jiawei/colorpeel-runs/color_material_composition_v1/20260924-195852__color_material_composition_v1__orange_token_local_color_short100__e606386__42`.
The training manifest SHA-256 is
`a0f4518a219c7b73803caac9ce09727ab2c7a76246382e6ca5514e8701715baf`;
it records `succeeded` and return code 0 after 100 steps. The color adapter
weights, adaptation metadata, and `<C*>` embedding SHA-256 values are,
respectively,
`50a7235b9d7730b61d2e88b93c1a18fc2af13c22a7a4da7011949f9024ebe6a8`,
`0aaffb3730fd34ab6d92599b738c8d1e4d29e4ef23c3a875fc4445f2f57aa8d0`,
and `007b4a5087e78e39531d5c5300414b9a46540fda5d107e0e2f260b6c7c24cf7b`.
The short run changes the color adapter architecture from the earlier shared
K/V branch; its success does not imply equivalent color transfer quality.
Its concepts file supplies no object mask, so the current training loss uses
the full image region. The YAML sets `hflip: true`, but the dataset's flip
transform is commented out in `src/train/train_colorpeel.py`; that setting
does not actually augment this run. Neither observation alone establishes a
cause for weak color transfer.

The 36-image evaluation ran under
`/home/r12user5/Documents/Jiawei/colorpeel-runs/color_material_composition_v1/20260924-201139__orange_metal_diagnostic_36__74a86cd__42`.
Its protocol SHA-256 is
`93c1b5127bccf7fdf262e0375ad1eb3e491403bdd85dbc36e2a1465f7cf0b96f`.
It compares base, color-only, material-only, and combined prompts for cube,
sphere, and mug at seeds 42–44, 100 steps, CFG 3.5, with the safety checker
enabled. `<C*>` and `<M*>` use separate position gates on two frozen K/V
residual adapters; no paired training or joint checkpoint was used. The
generator records both source directories and the protocol hash.

The contact-sheet bundler verified 36/36 unique manifest/status rows and all
image hashes, RGB modes, and 512×512 sizes. The manifest SHA-256 is
`c3c3fcbd78d11d76950a35409a95d8d0e118412ae4ea1b1d0976cd5d9dd10c3ef`;
the status ledger SHA-256 is
`7650dec7eb4590a627649b2b9960d110a5f820e16a08818077b1968cce2503159`.
There are 35 unfiltered images and one safety-filtered base sphere at seed 43.
The filtered result is missing visual evidence, not an attribute-quality
failure.

## Project-owner visual review and decision

The project owner reviewed all three contact sheets after Codex's initial
screening. This review supersedes the initial screening wherever they differ:

- Cube seed 44 does not produce a cube even in the base condition. Its failed
  paired output therefore cannot be attributed specifically to `<C*> + <M*>`.
  At seed 42 the paired cube lacks convincing metal; at seed 43 metal appears
  but orange does not.
- The color-only mug is not clearly orange at seeds 42 and 44; seed 43 has
  visible orange. Material-only mugs satisfy the metal target. In paired mugs,
  the orange color is unclear. The orange fruit associated with `<C*>` is a
  known but lower-priority issue for this review.
- The color-only sphere is orange at seed 42, but orange is weak at seeds 43
  and 44. Material-only spheres satisfy the metal target. Paired spheres have
  acceptable metal appearance but lean toward gold rather than the requested
  orange.

The owner review points to weak or inconsistent `<C*>` expression, including
when `<M*>` is absent, while `<M*>` is comparatively reliable in mugs and
spheres. This is an observation, not proof that the color checkpoint is
undertrained: the nine-image, 100-step color run, prompt wording, and
interaction between two adapters remain competing explanations. The diagnostic
**does not pass stable color-material disentanglement**. Preserve all samples,
including cube seed 44 and the safety-filtered base sphere, in the ledger.
The four conditions differ in prompt wording, and this small grid does not
support a quantitative generalization claim. Investigate color-only strength
and inference-time C/M interaction before changing the training objective.
The proposed staged ablations are recorded in `02_color_material_improvement_plan.md`.
