# Mailbox alpha-only diagnostic

Authorized 2026-11-05 after the soft-alpha pink preview changed shape. Start
with two controlled runs; no production default change or training job.

1. Replay the previous five-color hard-SAM pink (`6726dab`) with its exact RGB,
   binary alpha, fixed crop `[145,254,439,548]`, seed42 and pinned inference.
   Require actual conditioning and all native arrays to match before step2.
2. Copy that same full512 RGB byte-for-byte and replace only alpha with the
   saved BiRefNet component from soft-pink `87efe3a`. Force the same crop.
   Retain the official float32 RGB*alpha and PIL out-of-canvas black padding.

This bypasses only the automatic alpha-derived crop in the diagnostic CLI.
Production default preprocessing is unchanged. The new soft-alpha support
outside old SAM lies on old black RGB: the fixed-crop conditioning must equal
the version restricted to old SAM, checked exactly on CPU. Thus any changed
model input comes from alpha values inside the old support, not its expanded
boundary, new recoloring pixels, crop size or interpolation.

Run from a new clean GitHub checkout, after tests and GPU ownership check:

```sh
python -B -m experiments.mailbox_alpha_control_v1.run \
  --hard-run OLD_FIVE_COLOR_PINK_TRELLIS \
  --soft-run SAVED_BIREF_PINK_TRELLIS \
  --output NEW_EXTERNAL_RUN --gpu 0
```

Use identical saved-reference and common centered cameras for geometry-only
comparison in `render.py`, alongside each run's separately fitted review.
Different native tensors alone prove different geometry, not worse geometry.
Results apply to this image and seed; no automatic acceptance or retries.

Completed: [RESULTS.md](RESULTS.md). Alpha replacement did not reproduce the
severe relative-structure failure. Two bounded follow-ups therefore completed
the soft-alpha RGB x crop comparison: old RGB/new crop and new RGB/old crop.
The existing new RGB/new crop result supplies the fourth cell, with its older
generation commit explicitly recorded. Four new meshes total; no further
seed, alpha, color or camera optimization sweep was performed.
