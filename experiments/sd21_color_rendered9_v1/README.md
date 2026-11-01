# SD2.1 Color: rendered background9, 800 updates

Use the accepted nine images from continuous gray studio rendering, not the
postprocessed constant-RGB128 variant. The source file index is SHA256 pinned.
Validate every source file, mask, caption and raw-render background equality.
Stage byte-identical RGB copies in cube/sphere/cylinder x views0/8/16 order.
Historical preview metadata and existing experiments remain unchanged.

Both original/fullrank token-local K/V and rank64/alpha64 token-local K/V learn
the C embedding from the pinned SD2.1-base512 model; no resume or old weights.
Keep initializer orange, seed42, fp32 training, batch/accumulation1/1, embedding
LR1e-5, KV LR1e-5 original/5e-5 rank64, constant LR, weight_decay0, cos_weight0,
whole valid-image loss (no supplied object-mask loss), fixed order, and inherited
hflip=true whose trainer operation is commented out. Only the dataset and dose
change relative to the earlier SD2.1 gray9 experiment.

Each method trains800 updates (88 complete9-image cycles + first8 images).
Save matching adapters and1024-dimensional embeddings at400/600/800. The shared
export auditor now reads the strictly verified recipe's max_train_steps rather
than assuming1000, preserving old recipe behavior.

After training, automatically run the ten historical Color-only prompts with
PNDM100, CFG6,512px,fp16, checker enabled, seeds30–34 inclusive. Each method
produces3x10x5=150 images, total300. No Subject/Material/Texture compositions.
Training and transfer results for both methods share one new experiment folder.

Deploy by local commit/push, then fresh server Git checkout. Use the existing
environment and independently verify data/configs before formal kickoff:

```sh
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
TPY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$TPY -B -u -m experiments.sd21_color_rendered9_v1.run --output "$NEW_CHECK" --gpus 2 3 --smoke --execute
$TPY -B -u -m experiments.sd21_color_rendered9_v1.run --output "$NEW_FORMAL" --gpus 2 3 --execute
```

The isolated2-update smoke checks are never resumed into formal training. Both
finite training-to-inference queues record train_outputs and then terminate.
Check available GPUs and disk first; do not terminate existing jobs or remove
old files. Perform bounded startup verification only; no continuing monitoring.
