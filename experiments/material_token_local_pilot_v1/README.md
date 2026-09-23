# Material token-local pilot v1

## Audit and implementation plan

The subject baseline already provides single-token `<S*>` embedding training,
token-gated delta K/V, a custom checkpoint loader, and a run launcher. The CLEVR
renderer already loads `MyMetal` and changes its group `Color` input. Its metal
shader lives inside the external `MyMetal.blend` asset: object roughness,
metallic, and specular parameters are not exposed as independent controls.
The pilot therefore fixes that exact asset and records its SHA-256, while a new
locked profile varies light placement/energy/color and camera azimuth. The
renderer records the profile, asset hashes, camera, lights, mask, and image for
each request. No existing study profile or dataset is edited.

The first target is **CLEVR native metal** (`MyMetal`), a controlled metallic
surface preset. `<M*>` is the sole learned token. The fixed training caption is
`a photo of an object made of <M*>`; the token-local subject recipe supplies the
optimizer and checkpoint format. The preview grid is 3 shapes × 2 colors × 3
lights × 2 views = 36 images. Following visual approval, the full grid adds
green and yellow for 72 images. The external CLEVR `properties.json` must
confirm the four nominal RGB values before rendering.

## Preview gate

Use fresh directories below `$COLORPEEL_RUN_ROOT/material_token_local_pilot_v1/`.
Run the planner from the repository root:

```bash
python -m src.methods.colorpeel_ice.prepare_material_token_local_pilot \
  --mode preview --output-dir "$PREVIEW_PLAN"
CUDA_VISIBLE_DEVICES=3 blender --background --python-exit-code 1 \
  --python scripts/methods/colorpeel_ice/render_clevr_multiview.py -- \
  --requests "$PREVIEW_PLAN/render_requests.jsonl" \
  --profile experiments/material_token_local_pilot_v1/configs/render_profile.json \
  --output-root "$PREVIEW_RENDER" \
  --properties-json "$CLEVR_ROOT/data/properties.json" \
  --base-scene-blendfile "$CLEVR_ROOT/data/base_scene.blend" \
  --shape-dir "$CLEVR_ROOT/data/shapes" --material-dir "$CLEVR_ROOT/data/materials"
```

Review all 36 rendered images using `preview_review_checklist.md`. Put a review
record at a new path with `verdict: "pass"`, a nonempty `reviewer`, `reviewed_at`, and
`renderer_realization_sha256` equal to the SHA-256 of the completed preview
render's `renderer_realization.jsonl`. The planner writes a pending template.
The full-grid planner and training staging validate that record, the 36 request
rows, the render contract, and every preview image/mask hash before proceeding.
The preview pass permits full-grid rendering and staging; standalone training
requires its own explicit authorization.

```bash
python -m src.methods.colorpeel_ice.prepare_material_token_local_pilot \
  --mode full --preview-root "$PREVIEW_RENDER" --review-record "$REVIEW_RECORD" \
  --output-dir "$FULL_PLAN"
```

Render the full plan with the same Blender command and a new output root. Stage
the completed full render with `--stage-training-from "$FULL_RENDER"`,
`--preview-root "$PREVIEW_RENDER"`, and `--review-record "$REVIEW_RECORD"`.
Staging copies images and paired object masks, validates their hashes, and
writes `concepts.json` with a one-item prompt list, plus a provenance manifest.
The tracked training YAML stays blocked pending separate training approval.
Any later authorized YAML must set `status: authorized_after_preview_review`
and `material_pilot_authorization` with `preview_root`, `review_record`,
`staging_root`, and `staging_provenance_sha256`. The launcher rechecks the
preview approval, the staged concepts and manifest, and all 72 image/mask
hashes before it creates a training run.

## Evaluation and scope

The standalone evaluation protocol fixes five seeds and four groups: material
reconstruction, color changes, unseen objects, and lighting changes. Record
color/shape/lighting leakage, material collapse, prompt suppression, object
fusion, and style leakage as separate outcomes. The generation script uses the
custom token-local UNet loader and a `<M*>` mask on the conditional CFG row.
Actual generation requires a `--checkpoint-lock` JSON with schema
`material_token_local_checkpoint_lock/v1`, the evaluation protocol SHA-256,
model directory, completed training `run_manifest_path` and its SHA-256,
`checkpoint_sha256` for weights/adaptation/token, and `base_model`. The generator
verifies that the checkpoint belongs to a successful standalone material run.
Subject/color checkpoints, AlignIT, joint training, texture tokens, and natural
image material extraction are outside this pilot.
