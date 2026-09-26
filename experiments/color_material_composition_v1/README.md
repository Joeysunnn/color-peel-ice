# Color and material composition diagnostic

The first target is the orange emission color branch already used by the
subject/color mainline. The project owner selected the ground-reflection
`<M*>` run in
`../material_token_local_pilot_v1/configs/selected_material_source.json`.
This experiment first retrains `<C*>` as a separate token-local K/V adapter
from the existing nine verified orange emission images. No material images
enter color training, and no paired color/material images are used.

The short color run keeps the earlier orange branch's data, initializer,
optimizer, seed, and 100-step budget. `cos_weight` is zero because token-local
training supports one modifier token. The changed adapter architecture means
this is a new diagnostic color branch, not a continuation of the old shared
K/V checkpoint. Check color-only transfer before drawing conclusions from
composition.

The completed orange short run is fixed by
`protocols/orange_metal_diagnostic_v1.json`. Its first comparison requests
36 images: cube, sphere, and mug; base, color-only, material-only, and paired
conditions; seeds 42–44. Every image uses 100 steps, CFG 3.5, and the enabled
safety checker. The generator records each image hash and safety status.

After color training, compose the two frozen adapters at inference with
separate `<C*>` and `<M*>` token-position masks. Compare base, color-only,
material-only, and paired prompts at identical seeds and sampling settings.
Record color, metallic appearance, object identity, safety filtering, extra
objects, and framing separately. The orange fruit artifact seen previously
must remain visible. This stage does not train a joint checkpoint.

The completed first diagnostic and its limitations are in
`reports/01_orange_metal_diagnostic.md`.

## Archived crossed CLEVR-color plan

The crossed CLEVR orange/blue × metal/rubber plan below is superseded for C/M
training. The first preview was red and the second was brown/gold, despite
color-socket calibration. No full render or training from this crossed plan
should run. The previews remain as evidence about native-material color drift.

### Historical joint C/M CAA ablation (do not run)

The literal-metal control first compares `orange color and metal material`
against `orange color and <M*> material` at the same cube/sphere/mug seeds
42–44. Its output is a diagnostic control, not a training source.

The joint experiment uses the selected ground-reflection scene and a new
72-image crossed grid: three training shapes × orange/blue × metal/rubber ×
three lights × two views. The four color/material cells for each shape and
view share a render seed. The orange emission branch supplies the target color
reference; the material socket is calibrated against rendered previews. Its
rendered metal color must be checked visually; the socket value alone does not
establish the final color.
The first 12 requests form a soft-front preview. Review all 12 with the
generated checklist, then make a separate pass record tied to the preview
manifest hash. Full rendering, staging, and training follow that review.

Run from the server repository after a local commit/push and server pull:

```bash
export COLORPEEL_JOINT_CM_ROOT="$COLORPEEL_RUN_ROOT/color_material_composition_v1/joint_cm_caa_ablation_v1"
python -m src.methods.colorpeel_ice.prepare_joint_color_material \
  --output-dir "$COLORPEEL_JOINT_CM_ROOT/plan"
CUDA_VISIBLE_DEVICES=3 blender --background --python-exit-code 1 \
  --python scripts/methods/colorpeel_ice/render_clevr_multiview.py -- \
  --requests "$COLORPEEL_JOINT_CM_ROOT/plan/render_requests.jsonl" \
  --profile "$COLORPEEL_JOINT_CM_ROOT/plan/render_profile.json" \
  --output-root "$COLORPEEL_JOINT_CM_ROOT/preview" --limit 12 \
  --properties-json "$CLEVR_ROOT/data/properties.json" \
  --base-scene-blendfile "$CLEVR_ROOT/data/base_scene.blend" \
  --shape-dir "$CLEVR_ROOT/data/shapes" --material-dir "$CLEVR_ROOT/data/materials"
```

After the preview passes, save `preview_review.json` under the same root with
`verdict: "pass"`, a reviewer and review time, and the SHA-256 of
`preview/renderer_realization.jsonl`. Render the full grid using the same
Blender command without `--limit 12` and with
`--output-root "$COLORPEEL_JOINT_CM_ROOT/full"`. Stage it:

```bash
python -m src.methods.colorpeel_ice.prepare_joint_color_material \
  --plan-dir "$COLORPEEL_JOINT_CM_ROOT/plan" \
  --preview-root "$COLORPEEL_JOINT_CM_ROOT/preview" \
  --review-record "$COLORPEEL_JOINT_CM_ROOT/preview_review.json" \
  --render-root "$COLORPEEL_JOINT_CM_ROOT/full" \
  --output-dir "$COLORPEEL_JOINT_CM_ROOT/staging"
```

Train two fresh shared Custom Diffusion K/V checkpoints from this same staged
grid, with identical seed, prompt, masks, optimizer, and 1,500 steps. The
configs differ only in run name and CAA weight (`0` versus `0.2`). CAA acts
on the orange-metal rows containing both learned tokens. It aligns their
attention maps; it is not a guarantee of independent color/material control.
These shared-K/V runs are compared with each other for the CAA effect. The
earlier independent token-local checkpoints remain external baselines because
their adapter architecture differs.

Use `scripts/launch/colorpeel_run.py` with
`configs/joint_cm_caa0_1500.yaml` and `configs/joint_cm_caa02_1500.yaml` in
distinct run directories. After each run succeeds,
`scripts/methods/colorpeel_ice/generate_joint_cm_evaluation.py` generates
the fixed 36-image base/C-only/M-only/C+M comparison for cube, sphere, and
held-out mug at seeds 42–44. Inspect color, metal appearance, shape, and
safety filtering separately at matched seeds.

## Active unpaired emission-C / fresh metal-M plan

The active plan keeps the nine verified emission orange images as the C source
and renders a fresh 72-image metal-only set under the selected ground-reflection
profile with new seeds. Its colors are literal red, blue, green, and yellow;
orange is absent from M training. Each training prompt contains exactly one
learned token. The 5,100-step shared-K/V schedule has 100 C steps and 5,000 M
steps, matching the earlier standalone step doses while spreading C steps
throughout training. CAA is zero because no sample contains both tokens.
Shared K/V still updates on both branches. A strict per-step guard restores
the embedding row of the token absent from that step after AdamW updates.

The first 12 fresh material images form a preview. The selected metal look,
ground reflection, four literal colors, shapes, and masks must pass review
before full rendering and staging. From the server checkout after local push
and server pull:

```bash
export COLORPEEL_UNPAIRED_CM_ROOT="$COLORPEEL_RUN_ROOT/color_material_composition_v1/unpaired_emission_m_v1"
python -m src.methods.colorpeel_ice.prepare_unpaired_emission_material \
  --output-dir "$COLORPEEL_UNPAIRED_CM_ROOT/plan"
CUDA_VISIBLE_DEVICES=3 blender --background --python-exit-code 1 \
  --python scripts/methods/colorpeel_ice/render_clevr_multiview.py -- \
  --requests "$COLORPEEL_UNPAIRED_CM_ROOT/plan/render_requests.jsonl" \
  --profile "$COLORPEEL_UNPAIRED_CM_ROOT/plan/render_profile.json" \
  --output-root "$COLORPEEL_UNPAIRED_CM_ROOT/preview" --limit 12 \
  --properties-json "$CLEVR_ROOT/data/properties.json" \
  --base-scene-blendfile "$CLEVR_ROOT/data/base_scene.blend" \
  --shape-dir "$CLEVR_ROOT/data/shapes" --material-dir "$CLEVR_ROOT/data/materials"
```

After review, render the same manifest without `--limit` into `full`, then
stage it with `--plan-dir`, `--preview-root`, `--review-record`, `--full-root`,
`--run-root "$COLORPEEL_RUN_ROOT"`, and `--output-dir .../staging`. The
launcher config is `configs/unpaired_emission_m_shared_kv_5100.yaml`; the
existing fixed 36-image C-only/M-only/C+M evaluation script accepts its
completed shared-K/V checkpoint. The joint output remains an extrapolation
because no training image contains both learned tokens.

The completed 5,100-step run and fixed 36-image evaluation are selected as a
provisional C+M baseline after the project owner's visual review. The exact
training checkpoint, evaluation ledger, and known limits are pinned in
`configs/selected_color_material_source.json`. Two evaluation images were
safety filtered; the selection is a qualitative decision, not a claim of
general disentanglement.
