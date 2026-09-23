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
Standalone 5,000-step training and evaluation have not begun.
