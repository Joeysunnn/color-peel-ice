# Generic-caption metal spoon material diagnostic

Date: 2026-10-04. This is a controlled comparison against the original
shape-specific-caption full-map metal-spoon `<M*>` run. All outputs are in
independent directories under
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_caption_ablation_v1/`.

## Training audit

The run
`20261004-160531__natural_material_caption_ablation_v1__metal_spoon_generic_caption_token_local_kv_5000__d2fc5a0__42`
finished with manifest `succeeded`, return code 0, all 5000 metric rows finite,
and five 1000-step snapshots. Its 72 image and 72 mask files are exactly the
original full-map metal-spoon training files by path and SHA-256. The three
`instance_prompt` values changed from `a photo of a {shape} made of <M*>` to
`a photo of an object made of <M*>`; all optimizer/model arguments, seed and
training code are the same. The `<M*>` embedding received a nonzero gradient
at every step; ordinary token rows and ordinary K/V paths remained unchanged.
Final K/V SHA-256:
`308362e57e4f6166a979c22a0bc6ddf49cc1824787a1beb79c53002682c032827`.
Final `<M*>` embedding SHA-256:
`03020b7a418d2bb39a781d009e34e24b6153c982afda07e3a4d9238fceaecf5`.

## Fixed-prompt transfer

The run `diagnostics/matched_shape_vs_generic_final_20261004` finished
80/80 rows: base, literal `polished stainless steel`, original shape-caption
token, and new generic-caption token, for cone/mailbox, both `a {object}`
and `a photo of a {object}`, seeds 42–46. Sampling used SD 1.4, PNDM 100,
CFG 3.5. A base-prompt probe was pixel-identical across checkpoints. The
original token had two safety-filtered cone images; the new token had none.

The new generic-caption token removes much of the old white-gray synthetic
studio look. It also makes mailbox images more structurally recognizable:
all five photo-prefixed mailbox seeds are mailbox-like, whereas the original
shape-caption token made a plain cylinder at seed 45. However, the generic
token does not reliably transfer to cone. Seeds 42/43/45 produce a white
cap or a round metallic disk/bowl, seed 44 a sideways taper, and only seed
46 a clearly upright pointed cone in either prompt form; that cone is red
like a traffic cone, not consistently metallic. The generic token's color,
reflectance and apparent material vary substantially between seeds. Thus
changing the caption alters geometry/style binding, but does not yield a
stable material concept or robust unseen-shape control. The literal control
itself sometimes misses cone, so individual images should be read with their
base/literal controls rather than as a standalone success count.

The four full-size sheets are in
`review/diagnostics/matched_shape_vs_generic_final_20261004/contact_sheets/`.
The server run keeps the original PNGs, prompts, hashes, safety flags and
`succeeded` provenance. The downloaded sheet archive SHA-256 matches the
server archive:
`6aa5cf329be322f6bfb01bb2dfabfbbb6ebf3fd2d08fcbd17f4095f77aee0421`.

## Checkpoint progression

The independent
`checkpoint_diagnostics_metal_spoon_generic_caption_20261004` run produced
30/30 cone/mailbox token images for steps 1000, 2000, 3000, 4000, 5000,
seeds 42–44. Each step uses its own K/V and embedding snapshot; step 5000
was verified tensor-equal to the final export. Two cone images were safety
filtered (step 2000 seed 42, step 3000 seed 44). Cone seed 42 is a white cap
at every visible step; seed 43 is a round disk/bowl; seed 44 becomes a
sideways tapered metallic object. No step gives reliable upright cone
transfer across these seeds. Mailbox remains recognizable over the five
steps, with no compelling early-step fix to the material variability.
Sheets are in
`review/checkpoint_diagnostics_metal_spoon_generic_caption_20261004/contact_sheets/`.

## Decision

The caption-only ablation is a partial improvement for mailbox identity and
scene leakage, but it does **not** solve material transfer to an unseen cone.
Do not replace the baseline checkpoint with it. The next isolated test should
change the training object scale while restoring the original shape-specific
caption and keeping SuperMat maps, renderer, lighting, views, 72-image split,
optimizer and evaluation fixed. The old CLEVR pilot used substantially smaller
foreground objects, so object size is a measurable remaining difference.
