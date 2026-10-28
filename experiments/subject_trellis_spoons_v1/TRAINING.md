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

## Continue after the spoon initializer rejection

The first queues completed both180 fullrank models, then stopped at rank64 step0:
the trainer's Subject LoRA validation still allowed only the `mailbox` initializer.
The fix adds `spoon` to that exact whitelist; other adapter, rank, frozen-base and
initializer restrictions remain intact. Tests execute the actual trainer guard.

The user authorized reuse without retraining the completed fullrank. Run
`experiments.subject_trellis_spoons_v1.continue_training --subject SUBJECT
--output NEW_CONTINUATION --gpu GPU --execute`, where NEW_CONTINUATION is a new
child of the original `SUBJECT/continuations/` directory. Old status/logs remain
historical and unchanged; the new continuation's `status.json` is authoritative
for the remaining jobs.

```text
ORIGINAL_ROOT/SUBJECT/
  all180/training/fullrank/       completed source, read-only, old revision
  all180/training/lora_r64/       original failed attempt, retained
  continuations/NEW_ID/
    all180/training/lora_r64/     fresh1000-step training, fixed revision
    all180/training_index.json   both real source revisions and immutable hashes
    all180/step500,step750,step1000/
    blue_green72/                both fresh methods and all checkpoint tests
    fullrank_reuse.json          old fullrank and15 snapshot-state file locks
    train_outputs/               exact remaining commands, logs, current status
```

Each continuation audits the old1000-step fullrank and all500/750/1000 paired
snapshots against the recorded original Git revision and source hashes. It then
trains only180 rank64, runs all three180 inference steps with both methods,
and performs the complete72 comparison using both methods. No180 fullrank job
is scheduled. New rank64 and72 runs start afresh from the frozen SD1.5 model,
not from partial failed checkpoints. Same subject, data, hyperparameters and
material/context/color probes are retained.

Before kickoff, the continuation preflight audits reuse, actually exports the
three old fullrank snapshots without editing them, checks new rank64 launcher
configuration, and plans the72 comparison. Startup verification must observe
real rank64 training losses past the formerly failing initializer guard. Stop
checking after startup; no monitor or automatic retry is installed.
