# Scheme 3 three-material extraction, training, and transfer

Run root on `research12`:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_threeway_v1/run_20261003_scheme3_threeway_v1/`

## Material extraction

Qwen3-VL supplied categorical surface and gloss priors. SuperMat supplied the
numerical median roughness (R) and metallicity (M) within a reviewed material
region. The VLM did not set either number. Albedo color, wood grain, and other
texture were excluded from the two-scalar material concept.

| Source | Region | VLM prior | R | M | Semantic gate |
| --- | --- | --- | ---: | ---: | --- |
| Painted mailbox | Painted lid | Coated/painted, semi-gloss | 0.352941 | 0.039216 | Accepted from prior Scheme 3 run |
| Metal spoon | Bowl interior | Bare metal, glossy | 0.247059 | 0.847059 | Accepted from prior Scheme 3 run |
| Wood spoon | Bowl interior | Wood, matte | 0.294118 | 0.007843 | **Conflict:** the matte prior expects R >= 0.45 |

The wood estimate was used unchanged as a provisional extraction, so the
experiment tests the requested direct Scheme 3 procedure despite that
semantic conflict. The source, masks, SuperMat maps, categorical prior,
canonical records, and hashes are in `source/`, `supermat_inputs/`,
`supermat_outputs/`, `semantic_prior/`, and `canonical/` below the run root.
The previously observed SuperMat-to-Cycles round-trip bias was not calibrated
away.

## Counterfactual data and learning

Each source R/M was held fixed across 96 Cycles/Principled BSDF renderings:
four shapes x four base colors x three lights x two views. Sphere, cube, and
cylinder gave 72 training images per material; cone was held out (24 images).
The three grids therefore contain 288 images, of which 216 were used for
training. The paired-grid verifier found the same 96 nonmaterial conditions,
matching metadata, and pixel-identical object masks across all three
materials; its record is `paired_counterfactual_summary.json`.

Three independent `<M*>` adapters used the original **token-local K/V**
method, not LoRA. Each used SD 1.4, initializer `material`, 5000 steps,
learning rate 1e-5, seed 42, 512 px, and captions of the form
`a photo of a {color} {shape} made of <M*>`. All three training manifests
report success and have a final checkpoint. Their run directories are siblings
of the run root under `natural_material_threeway_v1/`:

| Source | Training directory prefix |
| --- | --- |
| Mailbox | `20261004-002359__natural_material_threeway_v1__mailbox_token_local_kv_5000__62a863d__42` |
| Metal spoon | `20261004-002359__natural_material_threeway_v1__metal_spoon_token_local_kv_5000__62a863d__42` |
| Wood spoon | `20261004-002359__natural_material_threeway_v1__wood_spoon_token_local_kv_5000__62a863d__42` |

## Transfer test

For each adapter, the matched grid generated 72 images: cone and mailbox
under `base`, literal-material, and `<M*>` prompts (three colors x seeds
42-44), plus fixed balanced-aligned `<S*>` with and without `<M*>`.
Sampling was PNDM 100 steps, CFG 3.5. The exact prompts, seeds, image hashes,
and checkpoint hashes are in `transfer/{material}/` and
`appearance_audit/{material}/`. Qwen3-VL then annotated the **visible
appearance** blind to the intended source material. Counts below are
annotation counts, not measured physical R/M.

| Material | Arm | Valid / total | Target shape | Color | Clear metallic appearance |
| --- | --- | ---: | ---: | ---: | ---: |
| Mailbox | Base | 18/18 | 15 | 17 | 3 |
| Mailbox | Literal painted metal | 18/18 | 13 | 18 | 2 |
| Mailbox | `<M*>` | 18/18 | **2** | 18 | 0 |
| Metal spoon | Base | 18/18 | 15 | 17 | 3 |
| Metal spoon | Literal polished metal | 18/18 | 15 | 18 | 11 |
| Metal spoon | `<M*>` | **17/18** | **1** | 17 | 6 |
| Wood spoon | Base | 18/18 | 15 | 17 | 3 |
| Wood spoon | Literal wood | 18/18 | 16 | 17 | 2 |
| Wood spoon | `<M*>` | 18/18 | **4** | 18 | 0 |

One metal-spoon token image was safety filtered and excluded from appearance
counts. The strict JSON parser rejected two identical `<S*>`-only images per
material because the VLM returned `green` outside its permitted color enum;
those parse errors do not affect the material-only table. Complete arm and
gloss counts, validity flags, and SHA256 input hashes are in
`transfer_summary.json`, generated with `summarize_transfer.py`.

The contact sheets agree with the shape counts: material-only `<M*>` prompts
often produce simplified boxes or cylinders on a synthetic background where
the base and literal prompts show a recognizable mailbox. The metal token
adds some glossy highlights, but is less often classified clearly metallic
than the literal polished-metal prompt and usually loses object identity.
Mailbox and wood tokens are visually similar, which is plausible for their
nearby two-scalar R/M values; this experiment does not test wood grain.

With the fixed Subject checkpoint, all nine `<S*> + <M*>` outputs per material
were classified as mailboxes. Clear-metallic counts were 8/9 for mailbox
material and 9/9 for both spoon materials. However, the `<S*>`-only arm was
already classified clear metallic in 5/7 valid annotations, and visual
paired sheets show little reliable metal-versus-wood distinction after
composition. The material token also changes some background and details.
These observations do **not** establish independent R/M control by `<M*>`.

**Conclusion:** the full extraction-to-training pipeline ran successfully,
but this V1 token-local K/V training did **not** demonstrate clean material
transfer. The most prominent failure is shape leakage in material-only
prompts; subject composition preserves mailbox identity but does not isolate
the source material reliably. No quantitative SuperMat R/M estimate was made
on generated diffusion images because a consistent target-object material
mask is unavailable for the distorted outputs. VLM appearance labels cannot
substitute for that measurement. The wood semantic gate conflict and the
known SuperMat-to-Cycles bias further limit physical interpretation.

Local review sheets are in ignored `review_wood_prepared/`; original runs and
previous checkpoints were not modified.
