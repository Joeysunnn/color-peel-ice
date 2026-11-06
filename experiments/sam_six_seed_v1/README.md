# Six SAM instances: four conditions x four seeds

The six fixed targets are `image_000__mask_160`, `image_001__mask_008`,
`image_003__mask_207`, `image_009__mask_205`, `image_046__mask_097`, and
`image_081__mask_130`. The confirmed upstream protocol uses each target's GT
bounding box to prompt SAM. GT provides only the target box; it does not clip
or replace the resulting SAM mask. This is not Scheme A automatic discovery.
This runner consumes those new SAM semantic receipts and their failed-target
evidence; it does not extract masks, determine correspondence, or read GT masks.

Input is an external JSON with schema `sam_six_seed_inputs/v1`, `git` equal to
the current clean checkout's `git_info()`, and exactly six `objects` entries.
Each entry has `object_key`, absolute `source_receipt`, and
`source_receipt_sha256`. A failed mask target instead includes a nonempty
`blocked_reason`; its16 planned mesh slots are retained as blocked. Optional
manifest `source_locks` are also verified. The manifest and semantic receipts
are locked before running.

For each available object, invoke the existing four-condition preparation
with `--mask-kind SAM`: CPU hard preparation once, TRELLIS-environment soft
preparation once. There is no per-seed RGB, mask or BiRef re-inference. The four
conditions are `original__hard`, `pink__hard`, `original__soft`, `pink__soft`;
each condition's one immutable receipt is shared by seeds42,43,44,45. Hard alpha
is the supplied binary SAM mask. Soft alpha is freshly prepared from that
object's current SAM input; its algorithm, intermediate files and pinned model
provenance remain in the preparation receipts. Original/pink pre-alpha RGB is
shared across hard/soft, and final RGBA zeros RGB only where selected alpha is0.
After successful hard and soft preparation, compare each color's pre-alpha RGB
using PIL-decoded mode, dimensions and pixel bytes. Record PNG file hashes, but
allow different PNG encodings. A pixel mismatch blocks all eight soft mesh slots
and preserves the hard group; neither image is silently corrected. If hard
preparation failed, record paired RGB verification as unavailable; the standalone
soft group may proceed without claiming a successful hard/soft comparison.

There are96 fresh meshes and384 planned view renders. Generation uses pinned
TRELLIS1024 cascade settings and official automatic alpha cropping, with no
`--fixed-crop`. Verify each recorded seed and unchanged model, sampler, encoder,
precision, sparse backend and normalization configuration across all successful
candidates. Seed is the only allowed generation-setting difference. No selector,
quality-score gate, seed ranking, retry, old mesh fallback or training is used.

Each mesh retains its own fitted source reference camera plus the existing
yaw0/+90/-90 views (pitch0.4, radius2.8, FOV40). Only the three canonical columns
share fixed camera parameters; the reference column is independently fitted.
The unchanged renderer supplies flat-face Lambert clay, normal and mask views.

One serial queue runs per assigned physical GPU. A hard/soft preparation failure
blocks that group's two conditions at all four seeds, without blocking the other
alpha group. Each generation/render failure is isolated to its condition/seed.
All96 slots remain visible. Each object has four comparison PNGs at
`OBJECT/variants/CONDITION/comparison_clay.png`: rows42/43/44/45, four view columns,
and explicit failed/blocked placeholders. Root `index.html` shows all comparisons.
Individual native meshes live at `OBJECT/variants/CONDITION/seed_SEED/trellis/native_asset.pt`.

## Launch

Review/test locally, push the authorized fork, clone a fresh server checkout and
verify its full commit. Build the SAM input manifest under that same Git. Set
`COLORPEEL_RUN_ROOT` and `COLORPEEL_OUTPUT_ROOT` to
`/home/r12user5/Documents/Jiawei/colorpeel-runs`. Check GPU occupancy and ownership;
the reused `execute` checks idle status before every subprocess. Use existing
pinned cached models only. Only bounded startup verification is requested;
do not add a monitoring process or perform automatic quality assessment.

```sh
python -B -m experiments.sam_six_seed_v1.launch \
  --selection EXTERNAL_GT36_SELECTION_JSON --output FRESH_RUN --gpus 0 1 2 3
```

The launcher first runs pinned box-prompted SAM intake as a separate process,
which exits and releases its GPU before mesh work starts. It writes the six SAM
receipts to `FRESH_RUN/intake/inputs.json`, then invokes the runner below with
`FRESH_RUN/meshes` as output. The gallery is therefore `FRESH_RUN/meshes/index.html`.
The launcher itself has no monitoring loop. To run from already prepared inputs:

```sh
python -B -m experiments.sam_six_seed_v1.run \
  --inputs EXTERNAL_SAM_INPUTS_JSON --output FRESH_RUN --gpus 0 1 2 3
```
