# Spoon blue/green72,600-step Subject comparisons

The user authorized new independent metal_spoon and wood_spoon experiments.
Each stages exactly72 blue/green RGB/mask/caption triples from the frozen180-image
counterfactual source. Preserve all source hashes and source order. No3D sampling,
recoloring, mask/loss changes, shuffle, added original photos or data correction
is introduced in this dose comparison.

Both original/fullrank token-local K/V and rank64/alpha64 start afresh from the
frozen SD1.5 base. Change only the previous spoon recipe's total updates from1000
to600 and paired checkpoint plan from500/750/1000 to200/400/600. Keep the spoon
initializer, seed42, learning rates, optimizer, precision, mask loss and batch1.
Six hundred updates retain eight full72-image cycles plus the first24 rows;
per-image exposure is8 or9. Reports and staged authorization record this dose.

Use one new run root, with all files of each subject together:

```text
NEW_ROOT/
  metal_spoon/
    training_dataset/ training_plan/ train_outputs/
    training/fullrank/ training/lora_r64/
    training_index.json
    step200/ step400/ step600/
  wood_spoon/                    same layout
```

Each subject's finite queue trains fullrank then rank64, then runs the three
checkpoint tests. Every test uses the adapter and Subject embedding from the
same200/400/600 snapshot. Five material prompts and four independent context/color
probes, both methods and seeds42/43 give36 outputs per checkpoint,108 per subject,
216 overall. PreservePNDM100,CFG3.5,512px,fp16 and the same safety checker.
The nine probe labels are metallic,matte,glossy plastic,ceramic,wood,in the snow,
surrounded by flowers,in white color,in orange color. No seed44-48 extension.

Train only from a new clean server checkout after local test/commit/push and
server clone/pull. Use the existing colorpeel017 Python environment. Example:

```sh
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.subject_trellis_spoons600_v1.train_pipeline \
  --subject metal_spoon --output "$NEW_ROOT/metal_spoon" --gpu 0 --execute
```

Before kickoff, dry-run both method launchers and all three checkpoint inference
plans. CPU tests exercise actual snapshot export guards and final600-step paired
consistency, not only CLI parsing. Confirm processes and real finite startup
losses, then stop checking as requested. No monitor, automatic retry, resume or
old-result overwrite. Keep the dedicated checkout HEAD fixed through scheduled
training/inference. This is a new dose comparison, not a claim that600 steps
eliminate extra handles or guarantee learned disentanglement.
