# Mailbox subject–material counterfactual dataset

The user authorized dataset construction after reviewing the single TRELLIS.2
asset and explicitly removed color preservation as an acceptance requirement.
This experiment renders the existing native asset; it neither samples another
3D object nor trains a model. All historical inputs/results remain read-only.
The180-image construction completed; see [RESULTS.md](RESULTS.md) for the
executed commit, output directory, verification and material-response evidence.

## Fixed protocol

`protocol.json` freezes the asset/camera/HDRI hashes and the complete180-image
factorial: five uniform colors × M{0,1} × R{.15,.50,.85} × three source-near
views × two official HDRIs (studio/courtyard). Every color/view/light block
contains all six material states. Colors use renderer display-domain RGB
constants; they do not claim to match the historical recolor images.

The full original919,180-vertex/1,836,844-face geometry is shared across all
samples. The simplified GLB is not used. Official `MeshWithPbrMaterial` uses a
single texture-free opaque material, with no normal/bump/displacement maps.
This deliberately replaces **all** generated Base Color and reflectance maps,
removing baked metal highlights/shadows from the appearance path. Geometry
details remain; printed patterns or details existing only in the predicted
texture are discarded. This is synthetic geometry-based data, not a claim of
exact counterfactuals of the original photograph.

Each camera retains the estimated source pitch/roll and adds yaw−.25/0/+.25
radians. Centering and distance are computed once per view from all native
vertices with a12percent image margin. These views show the complete generated
asset and do not preserve the source photograph's existing bottom crop. No
material-specific alignment or mesh modification is performed.

The official PBR renderer uses512px,SSAA1,one depth layer (all surfaces opaque),
fixed seed42 for stochastic SSAO, and its own ACES/gamma transform once. Its IBL
backend is an approximation, not a calibrated BRDF measurement. M/R are physical
range parameters[0,1], not the signed learned MARBLE controls. Captions describe
metallic/non-metallic and roughness only; there are no wood/cloth/ceramic claims.

## Execute

Develop locally, CPU-check, commit and push. Clone/pull the same clean commit
into a dedicated server checkout before running. Check GPU occupancy first.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export CUDA_VISIBLE_DEVICES=1
/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python -u -B \
  -m experiments.subject_trellis_dataset_v1.build --output "$NEW_DATASET_RUN"
```

Output must be new, external and below the run root. Failed partial attempts
are retained; no automatic retry, training, monitoring or package installation.
Input hashes and clean code state are verified before and after execution.

## Output and checks

- `dataset/images/<id>/image.png`:180 actual512px RGB rendered candidates,
  common neutralRGB127 exterior. Each directory contains only one image.
- `dataset/masks/<id>/image.png`: matching binary render-derived masks.
- `dataset/manifest.json`: captions, full controls, geometry/pixel/map hashes
  and pending-review state; `concepts.json`: one image/mask directory and one
  caption per concept, matching the base training Dataset's input structure.
- `dataset/pairs.json`:30 six-state matched blocks and210 single-factor pairs
  (three metallic flips and four adjacent roughness changes per block), plus
  actual foreground image differences. It is a pairing index, not a split.
- `raw/`: unchanged renderer display outputs over black, before exterior
  replacement; `cutouts/`: matching RGBA with zero hidden RGB.
- `geometry/`: each view's geometry mask and native normal buffer.
- `maps/`: authoritative float32 Base Color/M/R/Alpha G-buffers per
  view/color/material, shared by both lights. PNGs are display images only.
- `review/`: six five-color × six-material contact sheets, all actual images
  and captions in HTML, and a caption CSV.
- `verification.json`, `protocol.json`, `cameras.json`, `runtime.json`,
  `repro_outputs/`: completeness, technical checks, provenance and limitations.

Hard checks cover all180 unique combinations, constant intended G-buffer
parameters, no clipping/nonfinite values, identical mask/normal buffers per
view, unchanged native geometry/input hashes, image/mask/caption matching,
neutral exterior, exact rendered foreground preservation and non-identical
images for single-factor interventions. Contrast statistics and clipping
fractions support visual review; they are not a perceptual acceptance score.

All candidates remain pending review. The base trainer does not enforce the
manifest's `training_eligible=False`; it must not be launched merely because
`concepts.json` exists. Existing MARBLE staging expects35/14 images and must
not be reused for these180. No random image split would establish independent
geometry generalization: all views share one generated object, and pairs share
nuisance factors. A later evaluation must document grouping and dependencies.
