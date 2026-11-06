# ICBench GT36: original/pink x hard/soft alpha

Reuse the externally hashed 36-object GT selection (`icbench_gt36_selection/v1`)
from29 images. Identity, source-image and GT-mask hashes are checked by the
existing GT36 validator. Each object receives four fresh seed42 meshes:
`original__hard`, `pink__hard`, `original__soft`, `pink__soft`.
There are144 planned meshes and576 planned view renders. No prior mesh or
SAM/BiRef asset is attached to the new inputs.

The canonical photograph and binary GT mask first produce a semantic input
receipt. The CPU hard preparation makes the two GT-binary-alpha receipts.
The TRELLIS-environment soft preparation performs fresh BiRef inference from
this object's current GT input and makes the two soft-alpha receipts. Both
groups retain the same pre-alpha original/pink RGB; only alpha changes between
hard and soft. Final RGBA sets RGB to zero only where its selected alpha is zero.
BiRefNet sees the original RGB cropped to the GT bounding box plus10% of its
longest side on each edge (rounded up, clipped to the image). Map its predicted
alpha back to512, then zero it outside GT while retaining predicted values
inside GT. No hole filling or opacity normalization is applied. Raw predictions,
the crop and constrained alpha are retained, including on preparation failure.
Model pins are retained in receipts. Pink uses the established deterministic
hue0 RGB transformation inside GT only.

Use the existing pinned TRELLIS1024 cascade, seed42 and official automatic crop
from each condition's alpha>204. Never pass `--fixed-crop`. Every generated mesh
is rendered unchanged using its own fitted source reference plus the existing
yaw0, yaw+90 and yaw-90 views (pitch0.4, radius2.8, FOV40). The existing flat-face
Lambert lighting and clay/normal/mask outputs are unchanged. Reference cameras
are independently fitted between variants; they are not a shared-camera or
calibrated comparison. No quality-score gate or mesh selection is applied.

One serial queue runs per requested GPU. Preparation failures block only their
two alpha-group variants; semantic-input failure blocks all four for that
object. Generation/render failure is isolated to its variant. All144 slots
remain recorded, including blocked and failed states; there are no retries.
Hard successes remain visible when soft preparation fails. No branch training,
SuperMat, initializer selection, S11 generation or monitoring process is run.

Each `image_NNN__mask_MMM/variants/VARIANT/` contains its new `trellis/` asset
and `geometry_views/comparison_clay.png` four-view sheet. The object's own
`comparison_clay.png` stacks the four variant rows into16 views, with explicit
failed/blocked placeholders. Root `index.html` displays these object sheets.
`status.json` records commands, receipt hashes, generation and completed-view
counts; technical completion does not claim geometry quality acceptance.

## Launch

Use reviewed/tested code pushed to the authorized fork, a fresh clean server
checkout, and verify its complete commit. Set both `COLORPEEL_RUN_ROOT` and
`COLORPEEL_OUTPUT_ROOT` to `/home/r12user5/Documents/Jiawei/colorpeel-runs`.
Check GPU occupancy and process ownership before launch; the reused `execute`
also checks idle status before each subprocess. All models use existing pinned
cached assets. Only bounded startup verification is requested.

```sh
python -B -m experiments.icbench_gt_alpha_four_v1.run \
  --selection EXTERNAL_GT36_SELECTION_JSON --output FRESH_RUN --gpus 0 1 2 3
```

Preparation CLI used by the runner:

```sh
python -B -m experiments.icbench_gt_alpha_four_v1.prepare \
  --source-receipt OBJECT/input/source_receipt.json --output OBJECT/hard_inputs
TRELLIS_PYTHON -B -m experiments.icbench_gt_alpha_four_v1.prepare \
  --source-receipt OBJECT/input/source_receipt.json --output OBJECT/soft_inputs --soft
```
