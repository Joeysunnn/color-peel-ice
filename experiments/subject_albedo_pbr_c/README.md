# Experiment C: Albedo editing and original-subject PBR readiness

This independent data-only study starts from the original orange mailbox
photograph used by SuperMat and Subject B. It reuses the existing official
SuperMat extraction after checking its input/output receipt chain, rather than
rerunning a GPU extractor with changed settings. No training or diffusion
inference is started by this study.

## Implemented boundary

The current repository has a Cycles PBR renderer for generic shapes, but it
does not have recovered mailbox geometry/normal, image-to-surface mapping,
matched camera or lighting. Single-image SuperMat supplies A/M/R, not these
missing quantities. Existing `natural_material_fullmaps_v1/render_grid.py`
uses Generated/BOX projection on generic geometry; its own `PROTOCOL.md`
calls this a single-view-to-3D approximation. Those renders cannot stand in
for the original mailbox.

Therefore this first C implementation deliberately produces **parameter maps
and visualizations only**. It emits no recomposed RGB, no training concepts,
no training images and no synthetic claims of physical material control.
`renderer_readiness.json` records the missing inputs. All thirty manifest
rows have `rgb_image=null`, `render_status=not_rendered` and
`training_eligible=false`. The intermediate-preview status is distinct from
a complete dataset. A successful parameter audit does not establish identity
preservation, material editing success or disentanglement in generated RGB.

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
