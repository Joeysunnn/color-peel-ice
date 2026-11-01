# SD2.1 Color gray9

User-authorized training on the nine accepted uniform-gray-background geometric
RGB images. Both original/fullrank token-local K/V and rank64/alpha64 token-local
K/V plus target embedding start from the pinned SD2.1-base512 bundle. No old
learned weights or Subject/Material/Texture combinations.

The old Color recipe remains: C token, initializer orange, seed42, fp32 training,
batch/accumulation1/1, embedding LR1e-5, KV LR1e-5 original/5e-5 rank64, constant
LR, weight_decay0, cos_weight0, fixed order, historical hflip=true (inherited
trainer flip is commented out). No supplied object masks: the whole valid-image
region is trained, as in the old Color branch. Masks only verify source pixels.
1000 updates, paired adapter/embedding snapshots400/600/800/1000.

Transfer retains exactly the ten historical Color prompts, PNDM100, CFG6.0,
512px, fp16, common checker, seeds30–40 inclusive. 10x11x4x2=880 images.
Each seed has a fresh device generator. Safety-filtered rows remain recorded.
Do not inherit the wood branches' CFG3.5 or add S+C prompts.

The new dataset copy and explicit authorization live under the experiment.
Historical preview training_eligible=false fields remain unchanged. Source
files/captions/order and the SD2.1 model are hash verified. Formal training is
not a base-only comparison: both the image construction and base changed.

Local edit/test/commit/push precede server Git clone. Use a new external output:

```sh
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
TPY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$TPY -B -u -m experiments.sd21_color_gray9_v1.run --output "$NEW_SMOKE" --gpus 2 3 --smoke --execute
$TPY -B -u -m experiments.sd21_color_gray9_v1.run --output "$NEW_FORMAL" --gpus 2 3 --execute
```

Smoke is two independent2-update training/export/one-image checks and is never
resumed into formal training. Require successful paired1024-dimensional exports
and inference first. Formal work has two finite training→all-checkpoint-inference
queues in one folder, then stops. Preserve old files and existing GPU jobs.
After verifying formal startup and finite initial loss, disconnect without
monitoring. Use /home for this run because /var/tmp has insufficient headroom.
