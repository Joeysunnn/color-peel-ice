# SD2.1 Mailbox Subject1500 + wood Texture600

Follow-up on2026-11-02: the user requested Subject1000 + Texture1000. Use the same
entrypoint with `--subject-step 1000 --texture-step 1000` and a fresh output root.
Both methods and all sampling parameters remain unchanged. Subject1000 is an
intermediate paired snapshot of the original1500-step training; Texture1000 is
the final paired snapshot of its1000-step training. Do not change training configs
or retrain. The default pair remains S1500/T600; only these two requested pairs
are accepted. Status, source audit, row IDs and review title record the selection.

User selected the existing wood-grain Texture600 checkpoint on2026-11-02.
Compose the reviewed eleven-image Mailbox Subject1500 with that independent
Texture16 branch, matching fullrank/fullrank and rank64/rank64. No training.

Source Subject: completed center11 run87d6e18; source Texture: completed SD2.1
wood branch run19049ce. This is **Mailbox identity plus wood grain**, not the
Mailbox Texture trained under SDXL. Both adapters and both embeddings come from
the same pinned SD2.1-base512, each with its own same-step snapshot.

Prompt: `a photo of <S*> mailbox with <T*> texture`.
Seeds30–40 match the current Mailbox test:11 images per method,22 total.
PNDM100,CFG3.5,512px,FP16,common checker,unchanged token-local dual-adapter gates.
The inherited second adapter slot named `material` carries `<T*>` only.

Validate completed training manifests, original configurations, model and data
hashes, export receipts and direct tensor equality with S1500/T600 snapshots.
Do not re-export or mutate source checkpoints or old status files. The current
center11 monitor must show completed without recovery, matching the verified
source paths; an unexpected recovery requires reviewing its new locations.

Run from a new committed checkout, with outputs outside both source experiments.
Only GPU0 is used, serially; SDXL's GPUs1/2/3 and checkouts remain reserved.

```bash
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -B -u \
  -m experiments.sd21_mailbox_st_v1.run --output "$NEW_ROOT/experiment" --execute
```

The new output contains source_audit.json, per-method progress/ledgers/PNG hashes,
actual11 Subject and16 Texture training references, and a two-method comparison
page. A successful run means all22 images were generated and source bytes stayed
unchanged, not that texture binds correctly or identity preservation is solved.
