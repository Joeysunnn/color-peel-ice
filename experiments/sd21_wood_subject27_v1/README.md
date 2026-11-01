# SD2.1 wood Subject27: original and rank64

Train the user-approved27 images from `subject_trellis_wood27_v1` from the same
frozen SD2.1-base512 model. Original/fullrank remains token-local K/V residuals
plus the target embedding; rank64 uses rank=alpha=64 token-local K/V LoRA plus
the embedding. Neither trains the full UNet. Do not resume the72-image runs.

Only the scientific `concepts_list` argument changes from the prior SD2.1 Subject
recipe. Keep initializer `spoon`, token `<S*>`, training seed42, batch/accum1/1,
embedding LR1e-5, original KV LR1e-5, rank64 KV LR5e-5, float32 training, existing
object-mask loss and augmentation, constant LR and fixed source order.

Each method trains1000 updates and saves400/600/800/1000. After training, the
finite queue audits all four snapshots, pairs each with its own embedding and
adapter, and runs the same nine probes at every checkpoint:

- metallic, matte, glossy plastic, ceramic, wood surface;
- in the snow, surrounded by flowers, in white color, in orange color.

Generation uses seeds30–60 inclusive, PNDM100, CFG3.5,512px,fp16 and the common
safety checker. Each method generates1116 images, **2232 total**. No literal sweep,
Material/Texture training or combinations are included in this experiment.

The input is24 studio endpoint renders plus the original photo and its two LCh
recolors. It is read-only and pinned by its original audit/file hashes. The old
dataset-only receipt and `training_eligible=False` fields remain historical
records of its creation. **The user's new explicit instruction authorizes this
training**, recorded in the new experiment's `authorization.json`; do not edit
the old dataset or relabel its old approval status. `inputs/` contains byte-equal
review copies, while both training configs read the same frozen source paths.

At1000 updates, the first frozen row receives38 exposures and each other row37.
The old72-image experiment used13/14 exposures. Keep this dose difference in the
comparison; no reshuffling or reweighting is introduced.

Deploy local edits/tests/commit/push, then a new clean server clone/pull:

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.sd21_wood_subject27_v1.run \
  --output "$NEW_EXPERIMENT_DIRECTORY" --gpus 2 3 --execute
```

Omit execute for a separate immutable dry-run directory. The experiment rejects
old output folders, occupied GPUs and insufficient disk space (24GiB minimum).
Both methods, copied inputs, configs, complete training states, stepwise paired
exports, inference, actual RGB/caption references and `train_outputs/` stay
under one new experiment folder. Status is recorded in `experiment.json`,
`gpu_N.json` and each method's logs/ledgers. Verify actual27-image training and
finite startup losses, then disconnect. No monitoring job is created.

The prior SD2.1 helpers retain their defaults. This study supplies an explicit
dataset/config contract to the existing paired exporter, exporter to the existing
probe runner, and dataset verifier/reporter to the existing finite queue. The
shared trainer, attention implementation, model loader and sampling math stay
unchanged. Startup success does not establish better identity or disentanglement.
