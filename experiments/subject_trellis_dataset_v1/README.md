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

## Authorized Subject training and material transfer

The user subsequently authorized training on all180 candidates with the previous
original/fullrank and rank64 recipes:1000 updates each, paired snapshots at
500/750/1000, then the same material-transfer tests. `train_pipeline` stages a
new byte-identical copy of all RGB/mask/caption triples in source order. Approval
is recorded only in that new staging; original pending-review receipts are not
edited. No original photographs, extra images or subset are added.

The two methods reuse Experiment B configuration builders and trainer, changing
only dataset paths and study/status metadata. Both use the frozen SD1.5 base,
`<S*>`, mailbox initializer,seed42,512px,batch1,accumulation1,constant learning
rate,embedding LR1e-5,mixed_precision=no,hflip=false,cos_weight0 and the same
object-mask loss. Fullrank token-local K/V LR is1e-5. Rank64/alpha64 token-local
K/V LR is5e-5. No new attention loss, optimizer change, shuffle or resume.

The original sequential loader performs five180-image cycles plus the first100
images at1000 updates. Each image receives five or six exposures; exact color,
view, light and material counts are written to `train_outputs/EXPOSURES.json`.
This changes exposure per image relative to the previous35/14-image datasets.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
TPY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$TPY -B -u -m experiments.subject_trellis_dataset_v1.train_pipeline \
  --candidates /var/tmp/colorpeel-r12user5/runs/subject_trellis_dataset_v1/20261027-192223__material180__0d7fce4__42/dataset \
  --output "$NEW_TRAIN_RUN" --gpu 1 --execute
```

Select the physical GPU after checking occupancy. Omit `--execute` for a separate
fresh dry-run staging/configuration plan. Existing directories are rejected.
Deploy code through local commit/push then a dedicated clean server checkout.
Keep that checkout's HEAD fixed until all scheduled work finishes.

The single job queue runs fullrank, rank64, then step500/750/1000 inference.
Each checkpoint tests metallic/matte/glossy plastic/ceramic/wood with seeds42/43,
PNDM100,CFG3.5,512px,fp16 and the existing safety checker:20 images per step,
60 total. Each adapter is exported with its same-step Subject embedding. Contact
sheets include all180 actual training RGB images and their complete captions.
Existing MARBLE inference defaults remain unchanged; explicit injected dataset
verification/configuration and reference-count hooks support this new study.

`training_plan/CONFIG_COMPARISON.json`, `training_dataset/dataset_audit.json`,
`approval.json`, `train_outputs/COMMANDS.*`, logs/status and each method's audit
record evidence. A failed stage stops the queue without automatic retry. Verify
the detached process and first finite training losses at startup, then stop
checking as requested. The scheduled inference belongs to the same job; no
monitor, periodic polling task or automatic notification is created.
