# Experiment C: Albedo editing and original-subject PBR readiness

This independent data-only study starts from the original orange mailbox
photograph used by SuperMat and Subject B. It reuses the existing official
SuperMat extraction after checking its input/output receipt chain, rather than
rerunning a GPU extractor with changed settings. Data-generation commands never
start training or diffusion inference; the separately authorized runner below
requires explicit execution.

The first server preview and its observed limits are recorded in
[`reports/01_parameter_preview.md`](reports/01_parameter_preview.md).
The subsequent fixed-view PBR dataset, numeric checks and remaining geometric
limitations are recorded in [`reports/02_fixed_view_pbr_dataset.md`](reports/02_fixed_view_pbr_dataset.md).

## Implemented boundary

The current repository has a Cycles PBR renderer for generic shapes, but it
does not have recovered mailbox geometry/normal, image-to-surface mapping,
matched camera or lighting. Single-image SuperMat supplies A/M/R, not these
missing quantities. Existing `natural_material_fullmaps_v1/render_grid.py`
uses Generated/BOX projection on generic geometry; its own `PROTOCOL.md`
calls this a single-view-to-3D approximation. Those renders cannot stand in
for the original mailbox.

The initial parameter-only C implementation deliberately produces **parameter maps
and visualizations only**. It emits no recomposed RGB, no training concepts,
no training images and no synthetic claims of physical material control.
`renderer_readiness.json` records the missing inputs. All thirty manifest
rows have `rgb_image=null`, `render_status=not_rendered` and
`training_eligible=false`. The intermediate-preview status is distinct from
a complete dataset. A successful parameter audit does not establish identity
preservation, material editing success or disentanglement in generated RGB.
The separate fixed-view Cycles extension below supplies an explicit manual
scene instead of claiming these missing quantities were recovered by SuperMat.

## Source and edits

- `natural_texture_material_v1.extract.inspect_sources` verifies the existing
  full-map run, upstream canonical records, SuperMat inference receipt and
  original unmodified A/M/R. No inpainted patch or grayscale Texture crop is
  used. The common verifier also checks the other two sources in that old
  three-source receipt; C edits only mailbox.
- The original JPEG, original binary mask and SuperMat RGBA are checked
  against their frozen hashes. The RGBA RGB must equal the entire decoded
  JPEG and alpha must equal the original mask. The small painted-lid
  measurement region is not an editing mask. B's repaired boundary is not
  substituted: pixels added by that repair were outside this SuperMat input.
- `natural_subject_recolor_pilot.recolor` edits the Albedo, using the existing
  fixed-hue, gamut-aware LCh method without modifying its algorithm. Pink,
  green, cyan, blue and purple retain B's corrected names and existing angles
  0,120,180,240,300 degrees. This is the same nominal palette, not a guarantee
  that those names describe the perceived colors after changing the source
  from RGB to estimated Albedo. All RGB outside the original mask is fixed;
  foreground Lab lightness and chroma are preserved subject to gamut and
  uint8 quantization. Nearly achromatic input cannot acquire strong color
  under a hue-only edit.
- M=0/1 and R=0.15/0.50/0.85 are explicit foreground interventions, not
  extracted or calibrated BRDF measurements. Outside-mask source parameters
  remain unchanged. The exact float32 NPY maps are authoritative; separate
  PNGs are quantized display previews (e.g. R=0.5 displays as128/255).
- All5x2x3=30 combinations reuse their respective A, M and R maps independently.
  Each row saves paths, hashes, values, numeric QC and a proposed conditioned
  caption. Proposed roughness words are an experiment plan; numeric R alone
  does not guarantee a material name or perceptual gloss under unknown light.

## Reproduce the data preview

Edit, test, commit and push locally; use an independent clean server checkout
of `codex/subject-experiment-c`. Do not switch or update Experiment B's active
checkout: B's automatic audits require its original revision.

```bash
export PYTHONDONTWRITEBYTECODE=1
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$PY -B -m experiments.subject_albedo_pbr_c.prepare --output "$NEW_C_PREVIEW"
```

The default old full-map source is frozen in `protocol.json`; it can also be
provided explicitly with `--source-fullmaps-run`. Output must be fresh and
outside the repository and all source directories. Old inputs are hashed
before and after the run. This command requires only CPU map-editing packages
already used by the project. It has no training or rendering execution flag.

Outputs include byte-identical originals under `source/`, edited `maps/`,
`parameter_manifest.jsonl`, source receipts/locks, output hashes, numeric
`albedo_qc.json`, `protocol.json`, readiness/status JSON and `index.html`.
Three labelled grids show original extraction, five edited Albedos and all
thirty parameter combinations. They explicitly distinguish maps from RGB
training images.

## Quality gates and remaining implementation

Confirmed numeric checks require unchanged mask-outside Albedo, no clipping,
foreground lightness p95 error at most0.5 Lab units, nonempty chromatic
support, finite unit-range parameter maps, exact float32 foreground
interventions and unchanged outside-map values. Failed edits remain visible
for audit but are marked rejected in the parameter manifest. These checks
are not identity or material quality scores for nonexistent rendered RGB.
SuperMat Albedo can retain shading, specular highlights and estimation errors;
all input maps still need visual review.

A feasible next step is to obtain/fit a mailbox mesh with the lid, slot,
handle and seams, match its camera/silhouette to the photo, and establish
image-to-UV correspondences for the extracted A/M/R. Estimate or explicitly
calibrate fixed lights/visibility and record those assumptions. Then reuse
the existing Cycles material graph (A as sRGB Base Color, R/M as linear data)
with that geometry, render transparent foreground and composite against the
unchanged original background. Define how original cast shadows are treated;
pixel-identical background and physically changed shadows cannot both be
assumed without such a policy. A validated same-view geometry/normal and
relighting pipeline is another option; a guessed normal or Lambertian A*shade
composite is not an equivalent BRDF renderer.

Before admitting any RGB to training, check fixed pose/mask/background,
silhouette alignment, internal mailbox landmarks/seams, clipping/degenerate
outputs and whether varying M or R produces a measurable plausible response.
Reject structural changes, duplicate no-effect edits and failed materials;
review all accepted samples and captions. Only then create per-image
CustomDiffusionDataset concepts with one actual RGB/mask/caption per entry.

Subject learning can then reuse B's original and rank64/alpha64 token-local
K/V recipes, seed42, 1000 updates and500/750/1000 paired snapshots, with only
the approved C dataset path changed. No extra loss or architecture is planned.
Keep fixed-seed B material prompts for metallic,matte,glossy plastic,ceramic,
wood (PNDM100,CFG3.5,512px,fp16,safety checker,seeds42/43). Add corresponding
color-only and crossed color/material prompts with the same sampling. Compare
identity, requested color and requested material separately; unseen material
generalization cannot be evaluated at this parameter-preview stage.

## Physical foreground rendering with an explicit manual geometry fit

`pbr.py` adds a separately versioned Cycles scene after the user's choice of
strict PBR. It constructs a photo-aligned mailbox heightfield with a curved
cylindrical body, cap transition and backplate. This is a manually fitted
single-view2.5D surface, not recovered true geometry, hidden structure or
measured source normals/lighting. It supports only the fixed source viewpoint.
Geometry uses mask/explicit coordinates, never Albedo brightness or M/R.

The five accepted Albedo files remain unchanged and are projected by fixed UVs.
Actual Principled BSDF/Cycles computes diffuse/specular transport with fixed
geometry, camera, area lights, environment, samples256,seed42 and exposure.
Only A selection and numeric Metallic/Roughness inputs vary. There is no
imagegen, learned material editor, per-image normalization or RGB tone edit.
An emission pass is used ONLY to verify image-to-UV correspondence, never as
training RGB. Actual clay, normal and depth passes diagnose the fitted geometry.

```bash
# Clean independent codex/subject-c-pbr-scene server checkout.
PREVIEW=/var/tmp/colorpeel-r12user5/runs/subject_albedo_pbr_c/20261020-153651__parameter_preview__ac92f46__42
$PY -B -m experiments.subject_albedo_pbr_c.pbr \
  --preview "$PREVIEW" --output "$NEW_SCENE_PREVIEW" --gpu 2
# After inspecting the scene and six green M/R cases, use a NEW full-grid run:
$PY -B -m experiments.subject_albedo_pbr_c.pbr \
  --preview "$PREVIEW" --output "$NEW_FULL_GRID" --gpu 2 --full-grid
```

Both runs preserve the source preview. They save the mesh/config/hash, `.blend`
scene, source locks, exact command, original scene-linear Combined EXRs,
geometry normal/depth passes, projection audit, full512px PNGs, captions and
contact sheet. Preview mode produces six green cases; full mode produces30.
Masked background composition uses linear premultiplied RGB/alpha, with the
shared repaired-mask-outside pixels copied exactly. Original background shadows stay
fixed rather than being recomputed, so this is physical foreground rendering
against a fixed photographic background, not recovery of the entire real scene.

UV p95 error, silhouette IoU, finite EXRs, core coverage and provenance are
checked. These numeric checks do not automatically approve the manually fitted
normals, internal components, boundary or material appearance. Outputs remain
training candidates pending visual review; no training starts. If the fitted
geometry or source Albedo fails those criteria, improve and revalidate it in a
new version rather than labelling poor candidates as usable training data.

After reviewing every full-grid RGB, save a separate review JSON with
`candidate_manifest_sha256`, `reviewer`, thirty `decisions` (`id`, `accepted`,
`reasons`) and explicit `limitations`. Staging refuses missing/rejected
decisions, changed RGB/EXRs/scenes, more than1% clipped foreground pixels,
incomplete foreground coverage or changed background. Both rendering and
staging must use the same clean revision. It copies accepted RGB/masks into
fresh per-image asset directories, with the frozen captions and shared
Subject dataset format; source candidates remain unchanged.

```bash
$PY -B -m experiments.subject_albedo_pbr_c.stage_pbr \
  --candidates "$NEW_FULL_GRID" --review "$SEPARATE_VISUAL_REVIEW_JSON" \
  --output "$NEW_ACCEPTED_DATASET"
```

The accepted dataset's contact sheet shows the actual thirty training RGB
images and references the original, clay and normal diagnostics separately.
No training command is run by this stage. The final lighting is globally
calibrated against all five Albedos; illumination, display transform and
256 samples with adaptive sampling disabled stay identical across all30
conditions, with no per-image exposure adjustment.

The initial SuperMat alpha misses a strip at the cropped object bottom,
which otherwise retains original orange photo pixels. The B border-repair mask
was rejected for C after full-size inspection found background-post pixels in
its additions. Final rendering instead preserves the original42269-pixel mask
and completes only the bottom cut with the explicit fixed polygon in
`pbr_protocol.json`:682 added pixels,42951 total. Both source and derived masks
are hash-locked and the derivation is recomputed during staging. No original
asset or Albedo file is modified. Original-mask-visible UVs stay unchanged;
only the added1.6% of foreground samples use fixed nearest-valid-source-pixel
UV continuation. This is explicit boundary extrapolation, not newly extracted
Albedo. A2px offscreen mesh strip avoids false transparency where the real
object continues below the photograph. For partially covered silhouette
pixels, a fixed background plate is completed inside the mask with TELEA3;
all visible outside-mask background pixels remain byte-identical. The plate
is hidden under opaque foreground and is never used as a training subject.

## Authorized C training and automatic material-prompt inference

`data.verify_dataset` reads and rechecks the accepted thirty-image dataset,
its actual RGB/mask pairs, full captions, five frozen Albedos and every original
source receipt. It never edits data. `training.build_config` reuses B's recipe
builder; only experiment metadata and dataset paths change. Shared training
preflight, trainer, parameter freezing, losses, optimizer and checkpoint audits
are reused. B's default configuration and execution are unchanged.

Both methods run1000 updates, seed42, batch1,512px, no mixed precision, constant
LR, no augmentation; token LR=1e-5, K/V LR=1e-5(original) or5e-5(rank64),
rank=alpha=64 for LoRA. Paired Token/KV state is saved at500/750/1000. Thirty
images replace B's ten pairs, so fixed updates imply different per-image exposure.

```bash
DATA=/var/tmp/colorpeel-r12user5/runs/subject_albedo_pbr_c/20261020-210555__pbr30_dataset__8c639fe__42
# Inspect all configuration differences and commands without GPU training:
$PY -B -m experiments.subject_albedo_pbr_c.run \
  --dataset "$DATA" --output "$FRESH_PLAN" --gpu 2
# Each method also supports a real shared-launcher preflight (no execute flag):
$PY -B -m experiments.subject_albedo_pbr_c.training \
  --dataset "$DATA" --output "$FRESH_PREFLIGHT" --method lora_r64 --gpu 2
# Full detached kickoff; select another fresh output and keep this checkout fixed:
nohup "$PY" -u -B -m experiments.subject_albedo_pbr_c.run \
  --dataset "$DATA" --output "$FRESH_RUN" --gpu 2 --execute \
  > "$FRESH_RUN.launch.log" 2>&1 < /dev/null &
```

One isolated GPU trains original then rank64/alpha64, audits both and then
automatically generates `step500/`, `step750/`, `step1000/`. Each contains20
images (two methods, five material prompts, seeds42/43), full provenance,
generation receipts and three contact sheets with all30 actual training
images and their full captions. Prompt rows and paired snapshot export are
directly reused from B. Sampling stays PNDM100, CFG3.5,512px,fp16 with safety
checker. The five exact prompts are `a photo of <S*> mailbox with a {material}
surface` for metallic, matte, glossy plastic, ceramic and wood. This entrypoint
does not regenerate an old baseline or alter any old result directory.

`train_outputs/` records commands, logs, scientific changes, comparability,
summary and status. Training indexes are merged only after both completed
audits; any failure stops downstream work and preserves partial results.
Kickoff verification is not evidence of completed training or disentanglement.
