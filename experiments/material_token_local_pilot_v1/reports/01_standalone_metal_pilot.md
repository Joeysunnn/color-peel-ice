# Standalone metal token-local pilot: audit and preview gate

## Hypothesis and locked contrast

With CLEVR native `MyMetal` fixed across varying shape, base color, light, and
viewpoint, single-token token-local K/V can learn a transferable surface factor
`<M*>`. The first contrast is the same object and color under three named
lights and two views. Later evaluation changes one factor at a time: color on
spheres, object noun on unseen semantic objects, and lighting on red spheres.
This is a controlled material-factor test, not a natural-image material claim.

## Read-only audit at `971869e`

- **Confirmed:** The subject training path supports exactly one modifier token,
  zero-initialized token-gated delta K/V, frozen ordinary-token K/V, and the
  `crossattn_kv` recipe. Single-token training requires `cos_weight=0` and a
  one-token initializer. `concepts.json` needs `instance_prompt` as a list.
- **Confirmed:** The CLEVR renderer changes the native material group's `Color`
  input. It can seed camera and light variation, but the baseline has no named
  three-light/two-view grid and does not expose object roughness, metallic, or
  specular values as parameters. The pilot adds its own locked profile.
- **Confirmed locally:** `properties.json` has all four proposed RGB values.
  Local `MyMetal.blend` SHA-256 is
  `e47b6f920aaa4f5db0306eed01c309e0fcd933fdf2a8b6032a5fe5012268eca6`.
  The renderer rehashes assets in its actual execution environment.
- **Pending:** Blender inspection of the `MyMetal` node tree and visual review
  of the preview images. The asset hash fixes the preset but does not identify
  numeric roughness or specular settings from the local audit.

## Execution record

- **Confirmed:** The 36-row deterministic preview request plan and review
  checklist were generated locally. The request file SHA-256 is
  `c94e646d2d7ad470eb77c6fe214882526639072fb16f27f2dec706d65d942bc4`.
- **Confirmed:** The renderer's `--validate-only` preflight accepted all 36
  requests against the local CLEVR assets. Its canonical request SHA-256 is
  `87d6f9199f5d45f11dec82069ee5f4ba861aeccf27c9467d72dc22a233980011`;
  the locked profile SHA-256 is
  `caefa8485ca280f6867fbccae82720a57e27332303224e2a207a5336c31570b5`.
- **Confirmed on research12:** `colorpeel017` passed all 7 pilot tests, and
  Blender 4.2.11 validated and rendered all 36 preview requests on GPU 3 at
  commit `ee06dce`. The isolated run is
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/material_token_local_pilot_v1/20260923-211902__material_token_local_pilot_v1__metal_preview_36__ee06dce__42`.
  `renderer_status.json` reports `succeeded` with 36/36 completed. The
  `renderer_realization.jsonl` SHA-256 is
  `e2811a09eb8c93d529d8ba46d4f56541d106ba59059529af5b414537869391bf`.
  An independent check matched all 144 image, object mask, background mask,
  and scene JSON hashes; all 36 object masks are nonempty and touch no image
  edge (foreground range 11,866–22,504 pixels).
- **Human feedback on the first preview:** Cube and cylinder were accepted;
  the sphere showed an unwanted upper/lower band. This preview is superseded
  and must not supply the approval record for the full grid.
- **Pending:** Human review of all 36 preview images, the 72-row full render,
  training, and evaluation. No material-learning result or success claim exists.
- **Gate:** A human pass tied to the completed preview realization hash is
  required before full-grid planning or staging. The tracked training config
  is blocked, and the launcher rejects it until a new reviewed config is made.

Success requires recognizable metal appearance, visible color changes without
loss of material, transfer to unseen object nouns, lighting-responsive
highlights, and no strong color/shape/background/style leakage. Record each
failure mode separately and preserve every rendered sample and status row.

## Corrected preview after sphere feedback

The sphere has one mesh and one material assignment. In a single-image
diagnostic, removing the ground from glossy reflection rays removed the band;
with the original unlit world, the sphere became too dark. The pilot-only
profile now uses `ground_visible_glossy: false`, world RGB `[0.2, 0.2, 0.2]`,
and world strength `0.5` for every shape. The native `MyMetal` asset and all
object colors remain fixed. Historical render profiles are unchanged.

At commit `86d16c3`, research12 pulled this change, passed all 7 pilot tests,
and rendered a fresh 36-image preview on GPU 3 under
`/home/r12user5/Documents/Jiawei/colorpeel-runs/material_token_local_pilot_v1/20260923-232251__material_token_local_pilot_v1__metal_preview_36__86d16c3__42`.
The locked profile SHA-256 is
`821f6d2e4be14f4ac43643e38d1609fe43bcf46eff1bce25311769c2aea062972`;
the completed `renderer_realization.jsonl` SHA-256 is
`8b2979903ca54a81b5ea48bb6ac646be073dce64abe07365cd244bc70cb6fcc6`.
The renderer reports `succeeded` with 36/36 completed. An independent check
matched all 144 artifact hashes and found all object masks nonempty and clear
of image edges (foreground range 11,866–22,504 pixels). The corrected
`preview_contact_sheet.png` shows no horizontal sphere band; red/blue and
lighting/viewpoint changes remain visible. Human approval of this *new*
preview was pending at that point; the later approval is recorded below.

## Approved full grid and training asset staging

The project owner accepted the corrected preview in the current Codex
conversation and requested the next step. A `pass` review record was written
for that 36-image realization only; its SHA-256 is
`12d709e99de1666aab75d838419878819e56c7606c623237cc40a3e96dffb8d3`.
The full-grid planner validated the preview, all 36 artifact hashes, and the
review record before issuing 72 requests. Its original status file used the
stale label `planned_pending_preview_approval` despite a valid nonnull
`preview_approval_sha256`; the planner label was corrected after this run.

At execution commit `30121e7`, Blender 4.2.11 on research12 GPU 3 rendered
72/72 images successfully. The isolated run root is
`/home/r12user5/Documents/Jiawei/colorpeel-runs/material_token_local_pilot_v1/20260923-233410__material_token_local_pilot_v1__metal_full_72__30121e7__42`.
The `renderer_realization.jsonl` SHA-256 is
`8bc0897788ac6daf02151226be17b2b5833064de29146e4a89ad7d886062e07a`.
The grid has 18 views per color and 24 per shape; all 72 object masks are
nonempty and clear of image edges (foreground range 11,866–22,504 pixels).
The `full_contact_sheet.png` visibly separates red, blue, green, and yellow;
the sphere has no upper/lower reflection band.

The staging step checked the preview approval, every full-render image, mask,
background mask, and scene JSON hash, plus every copied image and mask hash.
It produced 72 paired training images/masks and a one-item `concepts.json`
with the prompt `a photo of an object made of <M*>`. Its status is
`staged_pending_separate_training_authorization`; the
`staging_provenance.json` SHA-256 is
`346a08edaa48ebf94835234bdf0fc1b0a551e534dde85d7db72261c2b994044f`.
Standalone 5,000-step training and evaluation had not begun at this staging point.

## Authorized standalone training

The project owner separately authorized the 5,000-step run. The local
authorization config was pushed and the server pulled execution commit
`ceffc7f`. A launcher dry run revalidated the review record, staging
provenance, concepts, and all 72 staged image/mask hashes. The actual run is
`/home/r12user5/Documents/Jiawei/colorpeel-runs/material_token_local_pilot_v1/20260924-000114__material_token_local_pilot_v1__standalone_metal_token_local_kv_5000__ceffc7f__42`.
Its manifest records `succeeded`, return code 0, seed 42, and 5,000/5,000
steps on GPU 3. Checkpoints exist at steps 1,000 through 5,000. All 5,000
logged losses were finite; mean total loss was 0.38352 for the first 100
steps and 0.25532 for the last 100. This is optimization evidence, not a
material-quality metric. No validation metric or best checkpoint was selected.

The final token-local K/V weights SHA-256 is
`b947b94ab2b82d45a6ea45c681cee8d374b703ce1049c6f6d4e29fb04fa971b8`;
the `<M*>` embedding SHA-256 is
`4fef8d87f96b99a34dd80d9dad1c3642d4bf846ee48f9245fbdb3e92ef9c7b2e`.
The embedding audit found 5,000 nonzero-gradient steps for `<M*>` and zero
changed ordinary embedding rows. Structured training evidence is in the
run's `train_outputs/` directory.

## Locked evaluation and visual inspection

The fixed evaluation protocol (`bd136a8f379c65c9b512d366f80a7f9f1d62dd51b43f275f9d62fe092c6c57b2`)
requested 12 prompts at seeds 42–46: five seen-cube, 20 sphere-color,
20 unseen-object, and 15 sphere-lighting samples. Each used 100 denoising
steps and guidance 3.5. The evaluation was locked to the completed training
checkpoint by `checkpoint_lock.json` SHA-256
`e0f2bb0266cc205fbc06636f3955fb2a67cf893a5ff3e14b5390915303cf600b`.

The first run generated all 60 images but its status file called six black
safety-checker outputs `ok`. The generator was changed only to record the
pipeline's `nsfw_content_detected` flag and label these rows
`safety_filtered`. Research12 pulled commit `b04d92e` and reran the same
checkpoint and protocol, with the safety checker enabled, in
`/home/r12user5/Documents/Jiawei/colorpeel-runs/material_token_local_pilot_v1/20260924-105204__material_token_local_pilot_v1__metal_evaluation_60__b04d92e__42`.
All 60 image hashes are identical between runs. Every rerun image is 512×512
RGB and matches its recorded SHA-256; `evaluation_qc.json` records the checks.
There are 54 unfiltered images and six `safety_filtered` rows: green sphere seed
44; yellow sphere seeds 43, 45, and 46; soft studio red sphere seed 43; and
strong side-lit red sphere seed 46. These six are missing visual observations,
not negative material-quality scores.

Codex inspected the four contact sheets as a qualitative screening pass, not
as the protocol's human rating. In seen reconstruction, red glossy cubes appear
at some seeds, but seed 43 has two cubes, seed 45 crops the cube, and seed 46
resembles a red enclosure. All five blue-sphere outputs are blue and show
highlights. Green seed 46 loses much of its green color; of the two usable
yellow outputs, seed 42 looks relatively matte and seed 44 looks metallic.
Unseen mugs, vases, chairs, and mailboxes are often recognizable and reflective,
but material and color are inconsistent: mug seed 42 lacks a handle, vase seed
46 looks transparent, and chair/mailbox seeds include red and dark surfaces.
Lighting prompts change highlights in several samples, while seed 45 remains
severely cropped and some samples add a stand or other scene content.

This pilot therefore has visible examples compatible with a transferable
reflective appearance, alongside color, shape, and composition failures. It
does **not** establish material disentanglement: six planned samples are
filtered, no human rubric has been completed, and there is no matched
base-model or adapter-free control to attribute the appearance to `<M*>`.
The run preserves all samples, status rows, four contact sheets, and a blank
`human_review_template.csv` for a later human assessment.
