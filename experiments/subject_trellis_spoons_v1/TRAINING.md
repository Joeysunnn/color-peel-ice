# Spoon Subject training, 180 and blue/green72

The user authorized independent Subject token training for both spoon sources,
each with all180 images and its complete blue/green72 subset. Every comparison
contains original/fullrank and rank64/alpha64, 1000 updates each, with paired
adapter/Subject embedding snapshots at500/750/1000. There are eight training runs
in total, all starting from the frozen SD1.5 base; none resumes an older run.

Training hyperparameters, loss, masking, precision, optimizer, sequential
DataLoader and seed42 match the previous TRELLIS Subject recipes. The subject
initializer and prompt class noun change from `mailbox` to `spoon`. No source
material name is imposed on all examples. The two subjects have independent
tokens/weights, despite sharing the `<S*>` spelling in their separate runs.

The new output root has this structure:

```text
NEW_TRAINING_ROOT/
  metal_spoon/
    all180/
      training_dataset/ training_plan/ train_outputs/
      training/fullrank/ training/lora_r64/
      step500/ step750/ step1000/
    blue_green72/                 same layout
    train_outputs/                subject queue reports
  wood_spoon/                     same two comparisons and methods
```

One finite queue per subject runs all180, then blue_green72. Inside each
comparison the two methods train in order, followed by all three step tests.
Run subjects on separate available GPUs. No monitor or periodic status check is
started. A failed stage stops that subject queue without automatic retries.

Each checkpoint tests the same five surface prompts (metallic, matte, glossy
plastic, ceramic, wood), both methods and seeds42/43:20 images per step,60 per
180-image comparison. The72-image comparisons retain the prior four independent
probes (in the snow; surrounded by flowers; in white color; in orange color),
adding16 images per step:36 per step,108 overall. Total scheduled inference is
336 images across both subjects and doses. Sampling remainsPNDM100,CFG3.5,
512px,fp16 with the same safety checker. These are the existing training-pipeline
tests; no additional prompt-form or seed44-48 extension is scheduled.

Every comparison keeps byte-identical RGB/mask/caption copies, source hashes,
new training authorization, exposure counts and config differences. Source
candidate files and old experiment directories remain read-only. All180 uses
five full180-image cycles plus the first100 rows;72 uses thirteen full cycles
plus the first64 rows, retaining source order. This is different exposure dose
per image; no random independent-source split or material/identity score is
claimed.

Edit/test/commit/push locally, then clone/pull the same clean commit into a new
dedicated server checkout. Use the existing `colorpeel017` environment, not the
separate TRELLIS reconstruction environment. Example subject invocation:

```sh
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.subject_trellis_spoons_v1.train_pipeline \
  --subject metal_spoon --output "$NEW_TRAINING_ROOT/metal_spoon" --gpu 0 --execute
```

Before full kickoff, stage both sizes, check both method launcher dry-runs and
all checkpoint inference dry-runs in a separate fresh preflight directory.
After kickoff verify live processes and first finite losses, then stop checking.
Keep the server checkout HEAD fixed until both queues end; subsequent audits
and scheduled tests validate that same revision.
