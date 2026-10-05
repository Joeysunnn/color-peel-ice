# Matched material transfer and checkpoint diagnostics

Date: 2026-10-04. Existing datasets, checkpoints, and earlier transfer results
were not changed. The experiment code was committed in the server checkout as
`d2fc5a0`.

## Matched inference

For cone and mailbox, evaluate both `a {object} made of <M*>` and
`a photo of a {object} made of <M*>` at seeds 42–46, SD 1.4, PNDM 100 steps,
CFG 3.5, with the safety checker enabled. The old selected CLEVR metal
checkpoint and current full-map metal-spoon checkpoint produced 80 images in
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_fullmaps_v1/diagnostics/matched_old_clevr_vs_metal_spoon_20261004/`:
20 base, 20 literal polished-steel, 20 old token, and 20 new token. A base
prompt probe was pixel-identical across the two loaded checkpoints. The new
metal token had two safety-filtered cone images; the old token had none.

The old CLEVR token preserved an obviously pointed cone for multiple seeds
under both prompt variants, though it also made a long-nosed object or a
truncated cone in some seeds. The full-map metal-spoon token made capped
bottle/frustum or flat-topped cylinder shapes in nearly all visible cone
images. Removing `a photo of` did not recover a pointed cone. The matched
mailbox prompts show that both tokens can retain the broad mailbox category;
the new token more often smooths away construction and, with the photo prefix,
seed 45 becomes a plain dark cylinder. This is a real prompt-and-seed-matched
advantage for the old token on cone shape, but not a same-material causal
ablation: MyMetal, scene, source maps, caption and object scale still differ.

The current mailbox and wood-spoon tokens were then tested against the same
20 base prompts/seeds with their own literal controls. Each run has 60 rows
(20 base, 20 literal, 20 token), including a pixel-equal base checkpoint
probe. Server runs are `diagnostics/matched_mailbox_20261004/` and
`diagnostics/matched_wood_spoon_20261004/` beneath the above study root.
Mailbox-token cone outputs are predominantly flat-topped red barrels; wood
token cone outputs are wood-patterned cylinders for all visible tested seeds.
The mailbox token has two safety-filtered images; the wood token has four,
and the wood literal arm one. Under the photo prefix, wood-token mailbox
outputs are mainly simplified wooden boxes/barrels, while the literal
`light wood` prompt still yields recognizable mailboxes in several seeds.
Some base and literal cone outputs are themselves ambiguous, so filtered or
ambiguous images are not counted as shape failures.

Local contact sheets for review:

- `review/diagnostics/matched_old_clevr_vs_metal_spoon_20261004/`
- `review/diagnostics/matched_mailbox_20261004/`
- `review/diagnostics/matched_wood_spoon_20261004/`

Each server run contains original PNGs, prompts, seed and checkpoint hashes,
image hashes, safety flags, contact sheets and succeeded provenance.

## Checkpoint progression

The current full-map metal-spoon run was evaluated at steps 1000, 2000, 3000,
4000, and 5000 for `a photo of a {cone,mailbox} made of <M*>`, seeds 42–44:
30 rows, of which 24 were newly generated and the six final-step outputs were
reused only after hash verification. Each early checkpoint's K/V and `<M*>`
embedding came from the same Accelerate snapshot. Step 5000's 32 K/V tensors
and embedding row were exactly equal to the final export before inference.
The run is at
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_fullmaps_v1/checkpoint_diagnostics_metal_spoon_20261004/`.

Cone seed 42 is recognizably pointed at step 1000 and becomes a flat-topped
cylinder from step 2000 onward. Seed 44 is cylinder-like at every step.
Seed 43 was filtered at steps 1000, 4000, and 5000; the remaining two are
blunt frustums. Mailbox images at steps 1000–4000 retain more slots, doors
and scene detail; several step-5000 images look smoother and more synthetic,
though some remain mailboxes. The trend supports late training as an
**amplifier**, not a complete explanation. It does not justify selecting
step 1000 without matched material-quality scoring.

Local sheets: `review/checkpoint_diagnostics/cone.jpg` and
`review/checkpoint_diagnostics/mailbox.jpg`. The server run records every
image hash and safety flag; filtered black outputs are missing observations.

## First single-variable ablation now running

The first ablation keeps the exact 72 full-map metal-spoon image/mask files,
SuperMat maps, renderer outputs, token-local K/V method, initializer, model,
optimizer, 5000 steps and seed 42. Its only training-example change is that
the three shape-specific prompts become the old generic
`a photo of an object made of <M*>`. The preparation audit verified all 72
image/mask hashes and each shape's 24 examples. Its config SHA-256 is
`943f0ac1f4ad9d096d5654bf50db3523b6ee25262c61a960818afffb14bfeead3`.

The launcher dry-run succeeded. The independent training run was launched on
GPU 1 at
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_caption_ablation_v1/20261004-160531__natural_material_caption_ablation_v1__metal_spoon_generic_caption_token_local_kv_5000__d2fc5a0__42/`.
Its first manifest read recorded `running`; completion and transfer have not
been checked. Per the user's standing instruction, training will not be
monitored. After the user reports completion, compare its final and early
checkpoints with the same matched cone/mailbox prompts and seeds before
changing any other training factor.
