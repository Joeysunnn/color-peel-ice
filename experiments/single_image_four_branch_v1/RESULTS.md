# Original Mailbox photograph: Scheme A integration run

Execution date: 2026-11-04. Producer commit:
`da8e990a2291aa382c5ac57c86102b606ce5811c`.

## Execution evidence

Preparation completed for all four retained Scheme A instances, with 176 images
in total: S11/C9/M8/T16 per instance. All downloaded training images match their
recorded SHA256 values; none are missing. The Subject recipe is one original
photograph, two foreground recolors, and eight fresh TRELLIS renders.

Training and paired export **succeeded for all 16 selected object/branch jobs**
at the requested S1000/C150/M200/T400 schedule. Every object completed all four
branches. The final independent check counted exactly 7,000 ordered update
records, all with finite losses, and verified hashes for all 16 adapter/token
pairs (32 exported weight files). The export audit also checked finite tensors, SD2.1 adapter topology,
actual initializer copying, paired final snapshot equality and zero changes to
ordinary token embeddings. There were no failed or blocked branches and no
training reruns.

| Instance | Subject | Color | Material | Texture |
| --- | ---: | ---: | ---: | ---: |
| object_001 | 1000 | 150 | 200 | 400 |
| object_002 | 1000 | 150 | 200 | 400 |
| object_003 (mailbox) | 1000 | 150 | 200 | 400 |
| object_004 | 1000 | 150 | 200 | 400 |

Each cell is a completed and verified final step, not a quality score or selected
best checkpoint. No transfer/composition generation or user quality review was
performed in this training run.

The local and fresh server checkouts each passed the 77-test related suite.
The server producer checkout was clean and its complete commit matched before
preparation and training. GPUs 0–3 were idle before training. Unrelated processes
were left untouched. No models or external source revisions were downloaded.
Preparation and training use the same producer revision, seed 42 and SD2.1
token-local LoRA64 recipes. No earlier object weights or derived Mailbox data
were attached to this run.

The SSH connection was reset during the final checks; reconnecting confirmed the
independent `nohup` job had completed successfully. Final verification at
2026-11-04 11:56:28 UTC found the producer checkout still clean, no training driver
process and no GPU compute processes. The connection interruption did not require
a training restart.

## Instances and initialization

| Instance | Scheme A category | S / C / M / T initialization |
| --- | --- | --- |
| object_001 | number plate | tiles / blue / material / floral |
| object_002 | door region | door / brown / material / grain |
| object_003 | mailbox | box / orange / material / plain |
| object_004 | door region | pole / beige / glossy / striped |

The four predicted native masks reproduce the preceding Scheme A run. The two
door-region masks are narrow partial regions, not verified independent whole
objects. Each retained instance has separate source/mask receipts, initialization,
new geometry, SuperMat maps, datasets and training directories. Repeated modifier
token strings are scoped to each object's independent branch weights.

## Visual limitations observed before training

These observations are evidence of unresolved scientific quality, not user
acceptance. No threshold, seed, mesh or texture search followed the inspection.

- The mailbox's fresh original-color TRELLIS renders still show conspicuous
  geometric distortion. Centering does not repair it.
- Automatic Texture rectangles can contain object structure. The mailbox patch
  includes lid/rim boundaries and curved dark lines despite the `plain`
  initializer. This run is not equivalent to the historical uniform T data.
- The number-plate Texture patch includes digits, flower decoration and plate
  divisions; it is not isolated fine surface texture. Its aggregate Color target
  renders muted lavender even though the semantic initializer is `blue`.
- The first door region is a narrow visible strip. Its Subject reconstruction is
  correspondingly thin, and its Texture renders carry panel/band structure.
- The second door region is an image-edge strip with a very narrow Texture crop
  and a thin Subject reconstruction. Full-object completeness is unverified.
- Material values come from SuperMat estimates in automatic eroded ROIs. Their
  physical correctness and the branch outputs' quality have not been accepted.

This run is an engineering integration test, not an initializer-only comparison
to v1/v2 or evidence of strict disentanglement, background compliance or spatial
binding. It uses different masks, geometry and automatic ROI data from historical
center11 experiments. No best CFG, training step or checkpoint is selected.

## Locations

Server producer checkout (retain it for later step changes with these data):

```text
/home/r12user5/Documents/Jiawei/colorpeel-checkouts/single-image-four-da8e990
```

Server run root:

```text
/home/r12user5/Documents/Jiawei/colorpeel-runs/single_image_four_branch_v1/20261104__mailbox_A_full__da8e990__42
```

- `prepared/prepared.json`: immutable object/dataset index and command receipts.
- `prepared/objects/object_###/`: fresh inputs, initializers, TRELLIS, SuperMat,
  automatic ROI evidence and S/C/M/T image datasets.
- `prepare.log`: preparation driver log.
- `training/object_###/{subject,color,material,texture}/`: independent training
  configs, source audits, logs and verified checkpoint exports.
- `training/checkpoints.json`: final exact checkpoint/outcome index (`succeeded`).
- `training/train_outputs/`: completed standard training evidence bundle.
- `FINAL_VERIFICATION.json`: compact exact checkpoint paths, file hashes and
  completed-step checks for all 16 pairs.
- `train.log`: training queue driver log.

Each branch's final export is `ckpts/step1000`, `ckpts/step150`, `ckpts/step200`
or `ckpts/step400`, respectively. It contains `pytorch_lora_kv_weights.bin`, the
matching `<S*>.bin`, `<C*>.bin`, `<M*>.bin` or `<T*>.bin`, adaptation metadata and
a provenance receipt. The required base remains the pinned SD2.1 model recorded
in each receipt.

Local preview and evidence root:

```text
C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/single_image_four_branch_20261104
```

`preview/index.html` links all 176 training images, instance overlays and ROI
evidence. `preview/summary.json` records counts and download hashes;
`preview/VISUAL_REVIEW.md` records the visual limitations.
`EXECUTION.json` records startup intent/evidence; it is not a completion receipt.
`RESULT.json` records the final success and quality boundaries. `CHECKPOINTS.md`
and `CHECKPOINTS.json` provide a compact index of all 16 pairs. Downloaded logs,
configs and audit receipts are in `training_evidence/`; weights remain on the
server and no model/data/image/credential files were added to Git.

The command interface defaults to all branches, while `--branches` and `--steps`
allow a chosen subset and a new schedule. See [README.md](README.md). Reuse frozen
data only with the same clean producer Git and a fresh training output directory;
changing steps starts new weights from the base model, not checkpoint resume.
