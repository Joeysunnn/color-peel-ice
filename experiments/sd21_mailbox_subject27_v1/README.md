# SD2.1 Mailbox Subject27,600 updates

New experiment, with both original/fullrank and rank64/alpha64 token-local K/V
plus `<S*>` embedding. No full UNet finetuning, old learned weights or resume.
Base: pinned SD2.1-base512 with the established common safety checker.

## Data

Take24 byte-identical images/masks/captions from the new source-framed Mailbox72:
blue/green x M[0,1] x R[.15,.85] x three views x studio only, retaining source
manifest order. Append the original orange512 photo, then historical blue and
green LCh recolored originals. No rerendering, recentering, segmentation or
new recoloring. Orange JPEG is decoded to PNG without changing RGB pixels;
its source JPEG is also retained. Recolors are copied byte-for-byte.

The three photos keep the original photographic background/canvas and share
their historical paired mask `b4d119d...`, distinct from the TRELLIS geometry
masks and reconstruction-source mask `550c1e13...`. They have original-photo
lighting, no fabricated scalar M/R, and the historical caption form
`a photo of <S*> mailbox in {color} color made of glossy painted metal`.
All actual RGB/mask/caption pairs are under the new experiment/dataset;
input_review/ shows the27 actual training images.

User explicitly authorized this training and inclusion of orange original.
Historical dataset-only/held-out flags remain untouched; the new experiment
records authorization separately. Source metadata/images are hash-verified.

## Training and inference

- Both methods600 steps; snapshots200/400/600; initializer `mailbox`.
- Seed42,512px,float32,batch1/accum1,constant LR,existing mask loss and sample
  order/augmentation. Embedding LR1e-5, fullrank KV1e-5, rank64 KV5e-5.
- After each method finishes, audit/export matching adapter and token embedding
  for all three checkpoints, then run the historical nine Mailbox probes:
  metallic/matte/glossy plastic/ceramic/wood surface, snow, flowers, white, orange.
- PNDM100,CFG3.5,512px,fp16,checker; seed30-40 inclusive;99 images/checkpoint,
  297/method,594 total. No two-branch combinations or new prompt sweep.
- Fixed-order600 updates expose the first6 rows23 times and other21 rows22 times.

## Deployment

Local changes/tests/commit/push, then a fresh server clone/pull. Check GPUs and
space before running. Six complete snapshots require about12GiB by measurement;
the launcher requires20GiB free including margin for final weights/exports/images.
Never delete old results. Choose two currently idle GPUs.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$PY -B -u -m experiments.sd21_mailbox_subject27_v1.run --output "$NEW_ROOT/experiment" --gpus 2 3 --execute
```

Keep launch log/PID receipt inside NEW_ROOT. Omitting execute makes a separate
immutable dry-run directory. Each method performs the shared launcher dry-run
before formal training. Dataset path is inherited via COLORPEEL_MAILBOX27_DATASET.
Finite queues include paired checkpoint export and transfer; no monitoring job.
Each method includes standardized train_outputs/ records. Verify actual27-image
training and first finite losses, then disconnect without ongoing monitoring.
Source-crop geometry errors remain; successful startup is not quality acceptance.
