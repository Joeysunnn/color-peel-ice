# Ground-reflection training comparison

## Fixed contrast and source

The project owner requested retraining and inference with the first metal
preview, including the sphere's horizontal ground-reflection band, for
comparison with the corrected pilot. The first preview at `ee06dce` contained
36 red/blue images only; it was not a full training grid and had been rejected
for the main pilot because of the sphere band. The separate
`comparison_authorized` record states that known issue explicitly and is tied
to the original preview realization SHA-256
`e2811a09eb8c93d529d8ba46d4f56541d106ba59059529af5b414537869391bf`.
It is not a retrospective `pass` review.

To keep dataset size fixed, the comparison uses the exact original renderer
profile from `ee06dce`: profile SHA-256
`caefa8485ca280f6867fbccae82720a57e27332303224e2a207a5336c31570b5`.
Relative to the corrected profile, the original uses world RGB
`[0.05, 0.05, 0.05]`, leaves the base scene's world strength unchanged, and leaves
the ground visible to glossy reflection rays. The corrected profile uses
world RGB `[0.2, 0.2, 0.2]`, strength `0.5`, and hides the ground from glossy
rays. The native `MyMetal` asset, shapes, colors, three lighting conditions,
two views, request seeds, and masks remain locked. This is a comparison of
those *combined renderer-profile changes*, not an isolated ground-visibility
ablation.

## Dataset realization

At execution commit `30dc547`, research12 GPU 3 rendered a new 72-image
four-color grid under
`/home/r12user5/Documents/Jiawei/colorpeel-runs/material_token_local_pilot_v1/20260924-113702__material_token_local_pilot_v1__metal_ground_reflection_full_72__30dc547__42`.
Blender 4.2.11 reported 72/72 completed. The full realization SHA-256 is
`533b0b256097e1ab1d1cfe306dd8ef4ff8cb83fe06498e4cddc19e7269c18555`.
All 72 masks are nonempty (11,866–22,504 foreground pixels) and clear of the
image edge. The 36 red/blue requests reproduce the original preview's camera,
light, and scene metadata exactly. Their mask pixels are identical; JPEG
rendering has tiny pixel differences (largest per-image mean absolute channel
difference 0.00114 on a 0–255 scale). The full contact sheet retains the
sphere's ground-reflection band.

The 72 comparison requests match the corrected training requests after
excluding the renderer-profile hash, and all 72 paired mask pixel arrays are
identical. Staging validated every rendered image, mask, background mask, and
scene hash and produced 72 image/mask pairs. Its `staging_provenance.json`
SHA-256 is
`a89485b466861dae4826c051b6616ae70f4daec9a7b47053b277d1989476c866`.

## Training and evaluation

The project owner authorized a new full run. The comparison config at
`5adbef9` changes only the run variant, dataset paths, and comparison
authorization fields relative to the corrected run. All model, optimizer,
caption, mask, seed, batch, 5,000-step, and checkpoint settings are identical.
The launcher dry run revalidated the original preview, authorization record,
staging provenance, and all 72 staged image/mask hashes.

Training ran successfully on research12 GPU 3 under
`/home/r12user5/Documents/Jiawei/colorpeel-runs/material_token_local_pilot_v1/20260924-114730__material_token_local_pilot_v1__standalone_metal_ground_reflection_token_local_kv_5000__5adbef9__42`.
The run manifest (SHA-256
`8b96442f48ca8192616b6882df54b2a420580efe3ae6c06990a0d1b28b2c052e`)
records return code 0 and 5,000/5,000 steps. All 5,000 logged losses were
finite; mean total loss was 0.58204 over the first 100 steps and 0.43856 over
the last 100. This is optimization evidence, not a material-quality score.
The `<M*>` token had a nonzero gradient at all 5,000 steps, and no ordinary
embedding row changed. The final token-local K/V weights SHA-256 is
`9c102acedacde84a19f3a41bae57245ec8c156cd059b9d8b69755a908311f516`;
the `<M*>` embedding SHA-256 is
`72a4ef265032c3919816868d337f132b96564c5cd106f0bae4d5f7ba31f27fa5`.
Structured training evidence is in that run's `train_outputs/` directory.

Fixed-protocol inference ran under
`/home/r12user5/Documents/Jiawei/colorpeel-runs/material_token_local_pilot_v1/20260924-132148__material_token_local_pilot_v1__ground_reflection_evaluation_60__5adbef9__42`.
The evaluation uses the same 12 prompts, five seeds (42–46), 100 steps,
CFG 3.5, base model, float16 inference, GPU 3, and enabled safety checker as
the corrected run. The protocol SHA-256 is
`bd136a8f379c65c9b512d366f80a7f9f1d62dd51b43f275f9d62fe092c6c57b2`;
the checkpoint lock SHA-256 is
`9fd27f7d2854f3f96b3567545dcaa179ab40ef8e152bfb7bdd70d3be28a6eabb`.
All 60 images are 512×512 RGB and match their recorded hashes. There are 55
unfiltered results and five `safety_filtered` black outputs: yellow sphere
seeds 43, 44, and 46; unseen mug seed 45; and soft-studio red sphere seed 42.
The corrected run has 54 unfiltered and six filtered results. Filtered outputs
are missing visual observations, not negative material-quality scores. The
pairwise comparison summary SHA-256 is
`2c79d3f3b4aa126a0c47375d541008a4323ecd6c21c76411272855322c1e9cd6`.

The project owner reviewed the results and judged this original-profile
ground-reflection version better in appearance. This is the selected material
source for subsequent experiments. The comparison changes the full renderer
profile, so the preference does not isolate the ground reflection as the
causal factor. Qualitative inspection still finds color, shape, and framing
failures, including occasional missing color, extra objects, and cropped
objects. Neither 55 usable images nor the owner's appearance preference alone
establishes material disentanglement. The selected checkpoint and render
profile are recorded in `configs/selected_material_source.json`.
