# Automatic multi-object image intake

`prepare_from_image.py` accepts one photograph without a GT mask:

1. Pinned Qwen3-VL classifies visible entities as object/background/part/text,
   with evidence; only whole-object categories proceed to detection.
2. Grounding DINO locates instances; SAM independently segments every detection.
3. Mask overlap suppression removes duplicate and nested part candidates.
4. Existing controlled-v2 S/C/M/T initialization runs separately on each instance.

This is a source intake stage, **not yet a general image-to-checkpoint pipeline**.
The four-branch dataset resolver still requires the audited Mailbox SHA and is
not called here. Each new instance needs TRELLIS geometry, Color targets/renders,
SuperMat maps, Material/Texture ROI and derived datasets before training. A SAM
object mask is not an approved Material/Texture ROI or a mask for all rendered
training views. Existing Mailbox assets remain unchanged.

## Instance and coordinate contract

Discovery requests all identifiable whole-object categories, excluding parts and
background regions such as walls, floors and sky. Grounding determines instance
count; SAM does not name objects. Qwen/DINO can miss or misidentify objects.
Completion does not establish exhaustive discovery or mask quality.

Each box receives its highest-predicted-IoU SAM candidate. Masks across all
candidates with IoU >=0.9 or intersection/min(area) >=0.9 are deduplicated by
detection score; other overlaps are retained. The containment rule addresses
whole-object/part duplicates, but may retain a higher-scored part instead of its
whole object. It is a heuristic, not a guarantee of complete instance discovery.
Masks are never unioned. Scores, rejected duplicates and failures are recorded.
Predicted IoU is not measured accuracy.

`source.png` and full-image masks use EXIF-corrected native resolution without
resizing. Masks are binary L PNG (0/255). Each `cutout.png` is RGBA with zero RGB
outside the mask. `vlm_input.png` crops the mask bounding box with 5% padding and
replaces pixels outside the mask by RGB127, hiding other objects while preserving
the target foreground. This differs from historical unmasked Mailbox VLM input;
the original-image and crop hashes are both preserved. These crops are for VLM
isolation, not replacements for source photos in physical appearance estimation.
TRELLIS's 512px receipt still needs explicit paired RGB/mask preparation.

## Run

Use a reviewed commit, push, and fresh clean server checkout. Outputs must be in a
new directory outside Git under the configured run root:

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT="$COLORPEEL_RUN_ROOT"
export PYTHONDONTWRITEBYTECODE=1
/home/r12user5/miniforge3/envs/ice/bin/python -B \
  scripts/methods/colorpeel_ice/prepare_from_image.py \
  --image /absolute/path/input.jpg \
  --output "$COLORPEEL_RUN_ROOT/sam_multi_object_intake_v1/NEW_RUN" \
  --gpu 0
```

Defaults use existing `ice-vlm` and `ice` interpreters and cached Qwen/SD2.1 paths.
Overrides: `--vlm-python`, `--sam-python`, `--qwen-model-dir`, `--sd-model-dir`.
Subprocesses run sequentially, releasing model memory between stages. GPU
occupancy is checked before each stage; existing jobs are never stopped. No
package changes or downloads occur. SAM/DINO revisions match the existing cache:

- Grounding DINO tiny: `a2bb814dd30d776dcf7e30523b00659f4f141c71`.
- SAM ViT-base: `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`.

Individual stages: `--stage discover`, or
`--stage segment --categories-manifest .../categories.json`, in their respective
environments. Use `--device` for a worker's logical device and check GPU ownership
before invoking a standalone stage. The complete entry maps physical `--gpu` to
worker `cuda:0`.

## Outputs and boundaries

- `discovery/categories.json`: category list, source hashes, raw/model receipts.
- `segmentation/objects.json`: individual masks, overlays, cutouts, crops, scores.
- `initializers/object_001/initializers.json`: one independent four-concept
  manifest per instance, including controlled Material v2 evidence; not weights.
- `intake.json`, `status.json`, stage logs: Git revision, source locks, commands
  and each object's outcome.

Success is `ready_for_derived_data`, never `trained`. Partial segmentation keeps
valid instances and reports `partial`; initializer failures do not suppress other
instances. Any incomplete result returns a nonzero exit code. Empty detections
and model errors are preserved; there is no GT fallback, reroll or overwrite.

SAM's AutomaticMaskGenerator also supports unprompted candidate masks, but they
are not guaranteed to correspond one-to-one to whole objects. This entry uses
semantic discovery and box prompts. See the
[official SAM repository](https://github.com/facebookresearch/segment-anything).
