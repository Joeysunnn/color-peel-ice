# Original Mailbox Color: SD1.5 token-local training and S+C transfer

The user authorized SD1.5 Color training from the original Mailbox and confirmed
ten historical Color transfer prompts plus three Subject+Color prompts (plain,
snow, flowers), using seeds30–60 inclusive. The tracked original photograph is
D1GT81/130, the same source as the accepted orange Emission9 dataset; use that
source, not the pink recolored foreground used for Subject reconstruction.

## Data and training

`data.py` checks the original photograph, approved Lab target, historical render
receipts, staging hashes, captions and source order, then copies the nine RGB
Emission images into a fresh dataset. No new rendering, recoloring, masks or
Subject training images are introduced. Each concept has one actual RGB and its
original caption. Both methods train this identical dataset from the frozen
SD1.5 bundle, not from the SD1.4 Color checkpoint.

- `<C*>`, initializer `orange`, seed42, batch1, accumulation1,512px,fp32 training.
- original/fullrank token-local K/V: embedding LR1e-5, K/V LR1e-5.
- token-local rank64/alpha64: embedding LR1e-5, K/V LR5e-5.
- Each1000 updates, paired adapter/embedding snapshots at400/600/800/1000.
- Keep constantLR, AdamW settings and existing valid-image-region reconstruction
  loss; no supplied object masks or new loss. cos_weight0.
- Set weight_decay0 instead of the old Color0.01 so ordinary embeddings remain
  unchanged. Keep the historical hflip=true flag; the inherited trainer's flip
  operation is commented out, so no actual horizontal flips are introduced.
- The explicit checkpoint list overrides the checkpointing_steps200 fallback.
  Fixed source order gives111 nine-image cycles plus the first image.

`training_plan/LEGACY_COLOR_COMPARISON.json` records exact argument differences
against the historical token-local Color recipe. Changes include base, dose,
weight decay and the added LoRA method; this is not a base-model-only ablation.
The shared trainer's Color LoRA validation and export metadata explicitly admit
only `<C*>`/`orange` in token-local mode. Other concept guards remain unchanged.

## Transfer protocol

Preserve the ten Color prompt texts in the frozen historical transfer protocol,
including their original grammar. Add only:

```text
a photo of <S*> mailbox in <C*> color
a photo of <S*> mailbox in <C*> color in the snow
a photo of <S*> mailbox in <C*> color surrounded by flowers
```

PNDM100,CFG6.0,512px,fp16,safety checker; independent generator per image.
This retains earlier Color sampling. Earlier Subject-only tests usedCFG3.5;
all new Color and S+C conditions here useCFG6.0. Seeds30 through60 give31 seeds.
Each method/checkpoint has13×31=403 outputs; four checkpoints and two methods
give3224 outputs. Filtered samples remain marked and are not silently replaced.

Color-only conditions load only the same-step C embedding and adapter. Joint
conditions load S1000 plus that C snapshot, matching fullrank with fullrank and
rank64 with rank64. Each branch's own adapter and embedding must be paired.
The existing dual adapter implementation has historical internal slot names
subject/material; this experiment explicitly maps those to S/C, without adding
Material weights or prompts.

The Subject source is the completed TRELLIS blue/green72 Mailbox run
`20261027-204842__both_methods1000_transfer__b3b850d__42` under
`/var/tmp/colorpeel-r12user5/runs/subject_trellis_blue_green72_training_v1`.
`prepare_subject` reuses strict cross-revision audit in the original clean b3
checkout and exports its verified1000 pairs into the new experiment. Subject
is not retrained.1000 is a fixed comparison reference, not a best-step claim.

## Execute and inspect

Local edits/tests/commit/push must precede server clone/pull into a dedicated
clean checkout. Keep that checkout fixed until its two queues finish. Check
current free GPUs and disk first; do not change old jobs or results.

```sh
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.mailbox_color_sd15_v1.pipeline \
  --output "$NEW_EXTERNAL_ROOT" --gpus 0 2 --execute
```

The study parent must exist and the run directory must be new. Omitting
`--execute` prepares a separate immutable full preflight, including historical
Subject audit, both training launchers and all eight inference plans. Each
executed worker trains its method and then runs all four transfer checkpoints.
Failures stop the affected queue without automatic retry or resume. After
finite initial losses/process/output checks, stop checking as requested; no
monitor or automatic notification is created.

```text
NEW_EXTERNAL_ROOT/
  training_dataset/ training_plan/ subject_reference/
  training/fullrank/ training/lora_r64/
  step400/{fullrank,lora_r64}/       # same for600/800/1000
    images/ training_references/ index.html
    generation_manifest.jsonl generation_status.jsonl provenance.json
  fullrank.queue.log lora_r64.queue.log
  train_outputs/ status.json plan.json
```

Each method/checkpoint's HTML displays the nine actual training RGBs/captions
and links thirteen prompt pages, each with31 seed outputs. Preserve all images,
hashes, model/optimizer records and provenance; old sources are read-only.
Successful execution is not a color/identity/disentanglement quality verdict.
