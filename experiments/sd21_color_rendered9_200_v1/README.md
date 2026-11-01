# Rendered-background Color9: fresh200-step supplement

User-authorized fresh original/fullrank and rank64/alpha64 training from the
same pinned SD2.1-base512. The original800-step experiment has only400/600/800
snapshots. This rerun does not resume later weights or claim bit-identical
reconstruction of its missing historical200-step state.

Reuse exact nine RGBs, captions and cube/sphere/cylinder x views0/8/16 order;
same hash-pinned rendered backgrounds and bounded-L/fixed-a/b foreground.
Only max_train_steps and checkpoint_steps change from800 training to200/[200].
Keep C/orange, training seed42, fp32, batch/accum1, constant LR, embeddingLR1e-5,
KVLR original1e-5/rank64 5e-5, whole-valid-image loss and all inherited settings.
200 updates cover22 full cycles plus the first2 images.

Automatically export each method's paired step200 adapter and1024-dimensional
embedding, then run the unchanged ten Color prompts, seeds30–45 inclusive,
PNDM100/CFG6/512px/fp16/checker.160 images/method,320 total. No combinations.

New outputs only beneath the original experiment's `supplemental200/` folder.
Local edit/test/commit/push, then fresh server checkout. Existing source data,
weights, logs, statuses, HTML and outputs remain untouched. Do not switch an
active historical checkout. Finite training-to-inference queues stop by themselves;
perform only bounded startup verification, then disconnect without monitoring.

Two measured old snapshots occupy2,175,472,765 and1,825,693,180 bytes; paired
exports occupy102,309,719 and14,862,541 bytes. This one-snapshot supplement
requires10GiB free, covering the approximately4GB states plus final weights,
exports,320 images/reviews and reserve. Other experiment disk thresholds remain
unchanged. Reuse the successful same-data/method2-step check from the800-step
experiment; do not run extra training smoke jobs for this dose-only change.

```sh
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
TPY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
# Separate fresh paths under original experiment/supplemental200/<new run>:
$TPY -B -u -m experiments.sd21_color_rendered9_200_v1.run --output "$NEW_PREFLIGHT" --gpus 0 3
$TPY -B -u -m experiments.sd21_color_rendered9_200_v1.run --output "$NEW_FORMAL" --gpus 0 3 --execute
```
