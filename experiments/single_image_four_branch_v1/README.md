# Single photograph to independent object checkpoints

Default intake is **Scheme A**: Qwen whole-object category discovery, all Grounding
DINO detections, separate SAM masks and existing duplicate suppression. Each
retained instance has its own input/mask receipt, semantic initialization, fresh
derived datasets and independent SD2.1 token-local LoRA64 branches. Scheme B and
historical Mailbox assets are not selected automatically by this entry.

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

For an already identified object, C/M/T preparation can explicitly reuse its
Scheme A SAM instance with `--objects-json /absolute/path/segmentation/objects.json
--object-id object_003`. The input photograph must match the recorded original
SHA-256; canonical RGB, SAM mask and isolated crop are verified before any GPU
stage. Only that instance is prepared. Qwen initialization, SuperMat and derived
datasets run afresh under the current Git revision. The original segmentation
manifest and its source files remain hash-locked through the new input receipt.
This option supports C/M/T; an already accepted Subject is retained separately.

Explicit intermediate paired snapshots are supported, for example:

```bash
$PY -u -B scripts/methods/colorpeel_ice/learn_from_image.py \
  --image /absolute/path/input.jpg \
  --objects-json /absolute/path/segmentation/objects.json --object-id object_003 \
  --output "$COLORPEEL_RUN_ROOT/paper_experiments_v1/OBJECT/training/NEW_MT_RUN" \
  --branches material texture --steps material=200 texture=500 \
  --checkpoint-steps material=150,200 texture=300,500 --gpus 0 1
```

Checkpoint lists must be strictly increasing positive integers, include the final
step, and stay within the selected branch's maximum. Each saved step exports a
matched adapter/embedding pair and receipt. Use a new run directory inside the
same paper object root as its retained Subject checkpoints and inference results.

## Source and scientific contracts

- Preserve the full EXIF-corrected source canvas and each native SAM mask; pair
  letterbox them to 512 when needed. Original/recolored Subject photographs keep
  the scene; their reconstruction loss uses only the respective instance mask.
- Subject geometry is newly sampled from **original and pink** RGBA candidates
  sharing one soft alpha (pink: historical Lab/LCh hue 0). The semantic SAM
  mask remains binary and unchanged. A fresh pinned BiRefNet inference supplies
  separate soft alpha: complete nonzero-alpha connected regions are associated
  to SAM instances, preserving all selected opacity values and their halos.
  Ownership requires intersection divided by the smaller component/SAM area
  to reach 1%; weaker contacts are recorded and ignored only for ownership.
  A region with multiple owners, or less than 80% SAM coverage, fails Subject
  preparation explicitly; there is no hard-mask clipping or fallback.
  `trellis_alpha` preserves full-native RGBA then jointly resizes it with
  LANCZOS. `trellis_input/source_receipt.json` binds that matting receipt and
  the pink condition, whose alpha is byte-identical to the matting alpha.
  Canonical `mask_path`/`image512_path` remain the original SAM mask/photo;
  `conditioning_alpha_path` is used only for TRELLIS. Original photo anchors,
  Qwen initialization and C/M/T still use the original photograph. This applies
  to every instance. Exactly two seed42 meshes are retained under
  `trellis_candidates/{original,pink}`; one failure does not prevent trying the
  other. `trellis_selection/selection.json` records native checks, saved-camera
  opaque SAM silhouette IoU and normalized symmetric mean boundary distance.
  Among candidates passing the existing guards, maximize IoU minus distance;
  when both differences are within 0.01 and 0.005 respectively, use a recorded,
  per-instance seeded random choice. All-invalid blocks Subject. See the
  [selection protocol](../subject_dual_color_selection_v1/README.md).
  No extra color/seed search or automatic retry is performed. The chosen
  receipt/mesh pair is used directly; the other candidate remains available.
  Exact-white foreground pixels are retained as neutral white without changing
  either mask; an all-white condition is explicitly recorded as unchanged.
  The eight synthetic views preserve the exact fitted source camera and crop,
  with studio environment and blue/green x metallic 0/1 x roughness .15/.85. Both
  review (conditioning alpha) and actual-render (SAM) IoU must reach 0.80; new
  edge clipping is rejected. These checks block obvious mismatches, not certify 3D fidelity. Photo
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
- Qwen category discovery and initialization observe a separate copy capped at
  1024 pixels on its longest edge, preserving aspect ratio with LANCZOS and no
  crop/upscale. Smaller images keep identical pixels. Qwen generation receipts
  record original/observed sizes and the processor's actual `image_grid_thw`.
  The source photograph, native-resolution SAM input/masks and shared512 data
  preparation are unchanged by this observation-only memory bound.
- A failed object/branch remains failed or blocked while independent usable
  branches proceed. Technical completion does not establish full-object masks,
  geometry quality, four-attribute disentanglement or background compliance.

## Outputs

Preparation: discovery, instance masks/crops, isolated initialization receipts,
per-object soft-alpha and pink TRELLIS conditioning receipts and new asset, SuperMat maps, ROI
evidence and C/S/M/T dataset records. The prepared/object manifests record the
conditioning policy and receipt hash.
All artifacts have source hashes; arbitrary images cannot attach old Mailbox data.

Training: each `object_###/{branch}/` contains its immutable prepared recipe,
configuration, training log, actual initializer-copy audit and paired requested-step
adapter/embedding export under `ckpts/`. Root `checkpoints.json` indexes outcomes
and exact checkpoint paths. Every branch retains its own token `<S*>`, `<C*>`,
`<M*>` or `<T*>`; names are scoped by object directory. Combining different object
identities is a later inference decision, not an implicit checkpoint merge.

Only final-step checkpoints are saved by default. The standard training evidence
bundle is under `train_outputs/`. No automatic quality tuning, transfer-image
sweep or background monitor is scheduled.
