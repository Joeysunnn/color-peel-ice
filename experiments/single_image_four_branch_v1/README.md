# Single photograph to independent object checkpoints

Default intake is **Scheme A**: Qwen whole-object category discovery, all Grounding
DINO detections, separate SAM masks and existing duplicate suppression. Each
retained instance has its own input/mask receipt, semantic initialization, fresh
derived datasets and independent SD2.1 token-local LoRA64 branches. Scheme B and
historical Mailbox assets are not selected by this entry.

| Branch | Training images per object | Default updates |
| --- | --- | ---: |
| Subject | 1 original + 2 foreground LCh recolors + 8 fresh TRELLIS renders | 1000 |
| Color | 9 fresh shape/view renders with measured pixel Lab target | 150 |
| Material | 8 fresh shape/color/light renders using SuperMat scalar M/R | 200 |
| Texture | 16 fresh shape/color/light renders using grayscale SuperMat albedo | 400 |

The interface defaults to learning all four branches, with explicit branch and
step controls. This keeps a single-command first run while allowing later
single-attribute experiments. There is no GUI in this change. Preparation and
training are separate stages so new step schedules can reuse frozen data and
write a new experiment without overwriting prior checkpoints.

## Run

First use a tested, reviewed, pushed commit and a fresh clean server checkout.
Run with the existing `ice` Python; workers use the separate cached Qwen,
TRELLIS, SuperMat, Blender and `colorpeel017` environments. Optional
`--runtime-json` overrides the explicit runtime/path map; no credentials belong
in that file. No dependency installation or model download is performed.

```bash
export PYTHONDONTWRITEBYTECODE=1
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT="$COLORPEEL_RUN_ROOT"
PY=/home/r12user5/miniforge3/envs/ice/bin/python
$PY -u -B scripts/methods/colorpeel_ice/learn_from_image.py \
  --image /absolute/path/input.jpg --output "$COLORPEEL_RUN_ROOT/single_image_four_branch_v1/NEW_RUN" \
  --branches subject color material texture \
  --steps subject=1000 color=150 material=200 texture=400 --gpus 0 1 2 3
```

Omit `--branches` and `--steps` for those defaults. Preparation uses the first
listed GPU sequentially; independent training jobs are queued across the listed
GPUs. Occupied GPUs cause a stage failure, never process termination. Use only
GPUs checked as available to you. CPU-render-only or multi-host execution is not
part of this entry.

For a reviewable staged run, use `--stage prepare` with a fresh output directory.
It writes `prepared.json`. Then run:

```bash
$PY -u -B scripts/methods/colorpeel_ice/learn_from_image.py \
  --stage train --prepared /absolute/path/prepared.json \
  --output "$COLORPEEL_RUN_ROOT/single_image_four_branch_v1/NEW_TRAIN_RUN" \
  --branches subject color material texture \
  --steps subject=1000 color=150 material=200 texture=400 --gpus 0 1 2 3
```

Later, repeat the training command with another fresh output, e.g.
`--branches material --steps material=250 --gpus 2`. This starts fresh weights
from the same base and frozen data; it is not checkpoint resume. Preparation and
training require the same full Git revision and clean checkout. Use the retained
producer checkout when reusing its data. A branch absent from preparation must
be prepared before training it.

## Source and scientific contracts

- Preserve the full EXIF-corrected source canvas and each native SAM mask; pair
  letterbox them to 512 when needed. Original/recolored Subject photographs keep
  the scene; their reconstruction loss uses only the respective instance mask.
- Subject geometry is newly sampled from each instance's original RGBA input.
  The eight synthetic views use the fitted orientation, centered full geometry,
  studio environment and blue/green x metallic 0/1 x roughness .15/.85. Photo
  recolors use the established LCh blue240/green120 transformations. No historic
  pink mesh, reviewed center11 image or original TRELLIS material is reused.
- C/M/T use an automatically eroded object ROI. Material medians are estimates,
  not measurements; high-confidence controlled-v2 contradictions block only M.
  Texture uses an all-valid interior rectangle of the same object's albedo;
  near-constant albedo is explicitly recorded as uniform gray without amplified
  noise. Automatic ROI selection does not constitute human approval.
- Qwen initialization reads isolated instance crops. The opt-in selected-v1
  manifest retains the complete raw four-key response, while only selected
  valid branches are authorized for training. It clarifies relief versus albedo
  texture and allows at most one candidate-repair generation for invalid token
  choices. Unknown confidence is not forced to a guess. Both attempts and hashes
  are preserved. Old v1/v2 manifest behavior is unchanged.
- A failed object/branch remains failed or blocked while independent usable
  branches proceed. Technical completion does not establish full-object masks,
  geometry quality, four-attribute disentanglement or background compliance.

## Outputs

Preparation: discovery, instance masks/crops, isolated initialization receipts,
per-object TRELLIS asset, SuperMat maps, ROI evidence and C/S/M/T dataset records.
All artifacts have source hashes; arbitrary images cannot attach old Mailbox data.

Training: each `object_###/{branch}/` contains its immutable prepared recipe,
configuration, training log, actual initializer-copy audit and paired final-step
adapter/embedding export under `ckpts/`. Root `checkpoints.json` indexes outcomes
and exact checkpoint paths. Every branch retains its own token `<S*>`, `<C*>`,
`<M*>` or `<T*>`; names are scoped by object directory. Combining different object
identities is a later inference decision, not an implicit checkpoint merge.

Only final-step checkpoints are saved by default. The standard training evidence
bundle is under `train_outputs/`. No automatic quality tuning, transfer-image
sweep or background monitor is scheduled.
