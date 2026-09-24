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

## Visual screening and decision

These are Codex visual observations of the three contact sheets, not a formal
human rating. Combined sphere outputs at seeds 43 and 44 show orange/copper
surfaces with clear reflective highlights. Seed 42 remains strongly orange
but has weaker evidence of metal. Cube composition is inconsistent: seed 42
is orange with little obvious metal, seed 43 keeps a metallic cube but weak
orange color, and seed 44 breaks into a clustered structure rather than one
cube. Mug composition is likewise inconsistent: seed 42 looks orange but
ceramic, seed 43 looks silver metal with orange objects nearby, and seed 44
has a brown/orange surface with less clear metal identity. Color-only mugs at
seeds 42 and 43 visibly include orange fruit or slices, preserving the known
semantic leakage rather than hiding it.

The diagnostic therefore demonstrates that some `<C*>`/`<M*>` pairings are
possible, but it **does not pass stable color-material disentanglement**.
The four conditions differ in prompt wording, and the small 36-image grid
does not support a quantitative generalization claim. Keep the full outputs
and filtered row for later review. The next planned `<S*> + <M*>` diagnostic
can be investigated independently; this result must not be reported as a
successful C+M composition or used to justify a joint S+C+M training run.
