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
- **Visual observation, not approval:** The contact sheet shows distinct red
  and blue objects, lighting changes, and viewpoint changes. Sphere highlights
  are pronounced; cube and cylinder metal appearance still needs human review.
  The run stores `preview_contact_sheet.png` beside `render_preview/`.
- **Pending:** Human review of all 36 preview images, the 72-row full render,
  training, and evaluation. No material-learning result or success claim exists.
- **Gate:** A human pass tied to the completed preview realization hash is
  required before full-grid planning or staging. The tracked training config
  is blocked, and the launcher rejects it until a new reviewed config is made.

Success requires recognizable metal appearance, visible color changes without
loss of material, transfer to unseen object nouns, lighting-responsive
highlights, and no strong color/shape/background/style leakage. Record each
failure mode separately and preserve every rendered sample and status row.
