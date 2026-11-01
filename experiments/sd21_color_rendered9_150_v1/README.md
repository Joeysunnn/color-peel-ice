# Rendered-background Color9: fresh150 steps, snapshots100/150

User-authorized new original/fullrank and rank64/alpha64 training from the same
pinned SD2.1-base512. Keep the exact rendered9 source RGBs/captions/fixed order,
C/orange, training seed42,fp32,batch1/accum1,constantLR,embeddingLR1e-5,KVLR
original1e-5/rank64 5e-5,whole-valid-image loss and all inherited settings.
Only max_train_steps and checkpoint_steps change to150/[100,150]. No resume
from200/400/600/800 weights.150 updates=16 full9-image cycles plus first6 images.

Each method saves both complete100/150 states. After150 updates, audit/export
each snapshot's own1024-dimensional embedding and token-local adapter, then
run ten unchanged Color prompts x seeds30–45 inclusive x both checkpoints:
320 images/method,640 total. PNDM100/CFG6/512px/fp16/checker unchanged.

Use a fresh folder under the original experiment/supplemental150. Old source
data,checkpoints,logs,statuses,indices and images remain untouched. Deploy via
local commit/push and fresh server Git checkout; no server source editing.
The two finite queues train, infer all requested pairs, then exit. Only bounded
startup verification is performed; no ongoing monitor is created.

Disk budget: two fullrank snapshots at2,175,472,765 bytes each plus two rank64
snapshots at1,825,693,180 bytes each total8,002,331,890 bytes. Paired exports,
final adapters,640PNG images and review pages require additional space. Require
10GiB free before starting. Keep all old files; fail preflight if insufficient.
The previously successful same-data/method2-step runtime checks are reused;
this dose-only change does not start extra smoke training.

```sh
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
TPY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
# Distinct fresh paths under original experiment/supplemental150/<new run>:
$TPY -B -u -m experiments.sd21_color_rendered9_150_v1.run --output "$NEW_PREFLIGHT" --gpus 2 3
$TPY -B -u -m experiments.sd21_color_rendered9_150_v1.run --output "$NEW_FORMAL" --gpus 2 3 --execute
```
