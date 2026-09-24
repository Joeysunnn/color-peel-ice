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

The locked pilot profile hides the ground from glossy reflection rays while
keeping it visible to the camera and shadows, and supplies a uniform world
light. This avoids the horizontal upper/lower reflection boundary observed
on the sphere in the first preview. The same environment is applied to every
shape; `MyMetal` itself is unchanged. Review records must refer to the current
profile hash and its newly rendered 36-image preview.

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
The project owner separately authorized the standalone 5,000-step training
after this full grid was staged. The tracked training YAML is bound to that
specific run with `status: authorized_after_preview_review` and
`material_pilot_authorization` containing `preview_root`, `review_record`,
`staging_root`, and `staging_provenance_sha256`. The launcher rechecks the
preview approval, the staged concepts and manifest, and all 72 image/mask
hashes before it creates a training run.

## Evaluation and scope

The project owner also requested a separate comparison against the first
preview, whose sphere showed the ground reflection band. That preview had only
36 red/blue images and was not approved as the main pilot. The comparison uses
`material_token_local_pilot_v1_ground_reflection.json` and its locked original
render profile to make a separate 72-image grid. Its
`comparison_authorized` record acknowledges the known visual issue; it is not
a `pass` review for the main pilot. The comparison keeps training settings and
the 60-image evaluation protocol fixed, and writes to distinct run directories.

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

## Selected material source for later composition

The project owner preferred the ground-reflection comparison after its
5,000-step training and fixed 60-image inference. Use
`configs/selected_material_source.json` to identify the renderer profile,
training checkpoint, and evaluation evidence for subsequent material work.
The separate `reports/02_ground_reflection_comparison.md` records the paired
comparison and its limitations. This selection does not authorize or imply a
joint subject/color/material training result.
