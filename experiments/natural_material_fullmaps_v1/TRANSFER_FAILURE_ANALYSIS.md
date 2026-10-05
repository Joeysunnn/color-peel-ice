# Full-map material transfer versus the CLEVR material pilot

Audit date: 2026-10-04. This note compares completed runs; it does not change
checkpoints, renderer outputs, or training data.

The matched cone/mailbox inference proposed below has since been completed;
see `DIAGNOSTIC_RESULTS.md` for its results. The statement that the *original*
old evaluation had no cone prompt still holds.

## Runs and what was actually tested

- Old selected material checkpoint: `material_token_local_pilot_v1/20260924-114730__material_token_local_pilot_v1__standalone_metal_ground_reflection_token_local_kv_5000__5adbef9__42/checkpoints` under `/home/r12user5/Documents/Jiawei/colorpeel-runs/`. The user's cited `20260924-132148__...__ground_reflection_evaluation_60.../comparison_to_corrected` is an evaluation comparing this checkpoint with the earlier corrected-renderer checkpoint, not a comparison with the present natural-material experiments.
- Present checkpoints: `natural_material_fullmaps_v1/20261004-114311__natural_material_fullmaps_v1__{mailbox,metal_spoon,wood_spoon}_fullmaps_token_local_kv_5000__241c32a__42/checkpoints` under the same run root. Training data are under `natural_material_fullmaps_v1/run_20261004_fullmaps_threeway_lighting_v2/`.
- The old evaluation has **no cone prompt**. It tests `a mailbox made of <M*>` at seeds 42–46. The present evaluation tests `a photo of a cone made of <M*>` and `a photo of a mailbox made of <M*>` at seeds 42–44. Both use SD 1.4, 100 sampling steps and CFG 3.5. Therefore the cone observation has no matched old baseline, and even the mailbox images are not matched-prompt comparisons.
- In the old ground-reflection comparison sheet, all five mailbox outputs are recognizable as a mailbox category, but vary in construction, color, and apparent finish. The old report also records color, shape and framing failures elsewhere. This supports better category retention in those five images, not universal shape/material disentanglement.

## Training differences

| Dimension | Old selected CLEVR pilot | Current full-map three-source run |
| --- | --- | --- |
| Material source | Native CLEVR `MyMetal` shader; no source-natural-image SuperMat A/R/M maps or comparable numeric R/M | One reviewed natural image per material; SuperMat Albedo, Roughness and Metallic maps, box projected onto each shape |
| Materials | One synthetic metal | Painted mailbox coating, metal spoon, wood spoon; independently trained `<M*>` for each |
| Images per token | 72 = 3 shapes × 4 colors × 3 lights × 2 views | 72 = 3 shapes × 3 lights × 8 views; 24 held-out cone renders, not trained |
| Training shapes | Sphere, cube, cylinder, 24 images each | Sphere, cube, cylinder, 24 images each |
| Color | Red, blue, green, yellow; 18 images each | Source Albedo color fixed within each token: red / gray / wood color |
| Caption | `a photo of an object made of <M*>` for every image | `a photo of a {sphere,cube,cylinder} made of <M*>` |
| Initializer | `metal` | `material` |
| Scene | CLEVR gray ground/darker world, original ground reflection | Bright white studio world and gray ground, altered three-light profile |
| Apparent object area | Mask 11,866–22,504 pixels at 512² (about 4.5–8.6%) | Mask 32,011–70,463 pixels (about 12–27%) |
| Render samples | Cycles 512 | Cycles 256 |
| Learning | SD 1.4, original token-local K/V, one `<M*>` embedding, foreground masked diffusion MSE, 5000 steps, LR 1e-5, seed 42 | Same model, update mechanism, loss, steps, LR and seed |

The foreground loss divides by mask area, so larger objects do not simply
multiply loss magnitude. They do show the model a larger and more repeated
synthetic object appearance. The new run has **no numerical excess of cylinder
examples**. Its 24 sphere and 24 cube examples are visibly distinct in the
rendered training contact sheets.

The current full maps preserve source color and visible pattern by design. Their
R/M maps have narrow value ranges: median R/M approximately 0.353/0.043 for
mailbox, 0.251/0.847 for metal spoon, and 0.294/0.008 for wood spoon. Thus
adding full maps mainly adds a fixed source color/pattern rather than large
within-material R/M variation. This can encourage source appearance binding,
but it is **not sufficient to explain** the shape failure: the preceding
`natural_material_threeway_v1` used R/M-only renderings, four varied colors,
the same 3-shape training split, and the same token-local K/V method, yet its
reported material-only outputs also often collapsed into simplified boxes or
cylinders. Its VLM audit counted target shape in only 2/18 mailbox-material,
1/17 valid metal-spoon-material, and 4/18 wood-spoon-material outputs across
its tested cone/mailbox/color prompts. These are VLM appearance judgments,
not geometry measurements.

## Working interpretation and limits

The failure follows the newer natural-material/Principled-BSDF training and
evaluation pipeline across three sources and even when source Albedo is
discarded. The shared large centered primitives, studio scene, material
representation, prompts, and initializer are confounded. Current evidence
cannot assign a single cause, or say that the learned token specifically
memorized the 24 cylinder examples. Token-local K/V gates where its residual
is applied; the learned `<M*>` key and value can still change attention and
image layout. The masked reconstruction loss has no explicit requirement that
shape remain controlled by an unseen noun.

The old `MyMetal` metal appearance and scene differ substantially from the
new SuperMat-derived material appearance. Its favorable mailbox category
result is a useful reference, but its material shader is not a controlled
replacement for any of the three extracted natural materials. The base
`a photo of a cone` images are themselves variable across seeds, so the new
cone test needs base and literal-material controls, already present in its
contact sheets. `<S*> + <M*>` is deliberately outside this diagnosis because
the fixed Subject checkpoint visibly dominates those compositions.

## Next tests, in order

1. **Match inference before retraining.** Run old and current checkpoints on
   identical `a mailbox made of <M*>`, `a photo of a mailbox made of <M*>`,
   `a cone made of <M*>`, and `a photo of a cone made of <M*>`, seeds 42–46,
   100 steps, CFG 3.5, with corresponding base/literal controls. Blindly score
   target-object category, visible material, and synthetic scene carryover;
   exclude safety-filtered images from visual scores. This determines whether
   old cone transfer also fails and measures prompt-prefix sensitivity.
2. **Use existing full-map checkpoints at 1000, 2000, 3000, 4000 and 5000
   steps**, with the same matched prompts/seeds. If shape collapses late while
   source appearance improves, select by joint shape/material score rather
   than training loss alone. Equal 5000-step schedules do not imply equal
   effective overfitting when the data distributions differ.
3. **Run one training ablation at a time**, beginning with the metal-spoon
   source: first replace only shape-specific captions with the old generic
   `a photo of an object made of <M*>`; then separately test old-style object
   scale/scene and genuinely varied shape/placement. Keep the SuperMat maps,
   optimizer, 72-image count, seed, steps and evaluation fixed for each
   comparison. Preserve source Albedo if the desired material concept includes
   its color and pattern. Keep an unseen shape or object for selection.
4. **Only after a data/prompt baseline transfers**, consider constraining the
   `<M*>` edit or its inference strength. Changing to LoRA, a new loss, or
   several renderer parameters together would not identify this failure's
   cause. Do not infer a fix for `<S*> + <M*>` from material-only tests.

## Evidence pointers

- Old protocol: `experiments/material_token_local_pilot_v1/protocols/material_token_local_pilot_v1_evaluation.json`.
- Old dataset/training report: `experiments/material_token_local_pilot_v1/reports/02_ground_reflection_comparison.md` and `configs/train_token_local_kv_ground_reflection.yaml`.
- New extraction/render/training protocol: `experiments/natural_material_fullmaps_v1/PROTOCOL.md`, `write_train_configs.py`, `evaluate_transfer.py`, and `transfer_protocol.json`.
- New prior R/M-only failure: `experiments/natural_material_threeway_v1/RESULTS.md`.
- Local current review sheets: `experiments/natural_material_fullmaps_v1/review/transfer/{mailbox,metal_spoon,wood_spoon}/contact_sheets/`.
