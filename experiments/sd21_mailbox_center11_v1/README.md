# SD2.1 Mailbox center11

Use the eleven images reviewed and authorized on 2026-11-02: eight fixed-center
studio renders (blue/green x metallic0/1 x roughness0.15/0.85), followed by the
orange photograph and historical blue/green recolors. `approved11.json` pins
their order, exact image/mask bytes and captions. Copy from the completed
SD2.1 Mailbox27 dataset; no rerender, recrop, recolor or new segmentation.

Both methods start from the pinned SD2.1-base512, with no old learned weights:
fullrank token-local K/V plus Subject embedding, and rank64/alpha64 token-local
K/V LoRA plus Subject embedding. Each trains1500 steps, saving paired complete
states at500/1000/1500. Initializer mailbox; FP32, seed42, batch/accum1, fixed
order, original object-mask loss. Embedding LR1e-5, fullrank KV1e-5, LoRA KV5e-5;
constant scheduler, AdamW weight_decay0. No other science arguments change.

Each method completes training, audits and exports same-step adapter/token
pairs, then runs the nine historical Mailbox prompts at all three checkpoints.
Seeds30-40,PNDM100,CFG3.5,512px,FP16,common checker:297 images per method,594 total.
No composition. GPU0 is the only selected GPU; methods run serially. GPUs1/2/3
are reserved for the running SDXL queue even when briefly idle. The new source
checkout and outputs are separate; old code, environments and results stay intact.

At1500 steps the first four images have137 exposures and the other seven136:
1092 synthetic and408 photo exposures. This is not an exposure-matched27-image
ablation. Fixed center rendering does not repair existing geometry errors.

Deploy local tests and commit/push before creating a new server checkout.
The user supplied sufficient home capacity; do not requery disk space.
Use the existing colorpeel017 environment without package changes:

```bash
export PYTHONDONTWRITEBYTECODE=1
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$PY -B -u -m experiments.sd21_mailbox_center11_v1.run --output "$NEW_ROOT/experiment" --execute
```

Omit `--execute` for a separate immutable CPU preflight. The runner performs
both launcher dry-runs before execution and records standardized train_outputs.
After a finite startup check, leave the queue to finish without ongoing monitoring.
