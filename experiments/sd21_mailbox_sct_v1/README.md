# SD2.1 Mailbox S1000 + orange C150 + wood T1000

User-requested three-branch inference, without retraining. Reuse the reviewed
center11 Mailbox Subject1000 and historical wood Texture1000 from the preceding
S+T test. Add the completed rendered9 original-orange Color150 from commit
5f3389aad843ae0eb631cb2422941cef30abb9ff. Match fullrank/fullrank/fullrank or
rank64/rank64/rank64 and load each branch's same-step learned embedding.

Prompt: `a photo of <S*> mailbox in <C*> color with <T*> texture`.
Both methods use seeds30–40,512px,PNDM100,CFG3.5,FP16,safety checker enabled:
22 outputs. CFG3.5 follows the preceding composition tests; the historical
Color-only transfer used CFG6. No literal `orange` word is added as a substitute
for learned Color. No Material branch is loaded.

The old dual-adapter processor cannot support three branches. A fixed S/C/T
processor retains its base attention path and adds each branch's own K/V
residual only at its distinct token position, with zero gates on CFG negatives.
Fullrank deltas remain fullrank; LoRA rank64/alpha64 retains its original two
linear projections. All branch scales remain1. Tests check zero gates, each
slot, combined slots and CFG token masks before actual inference.

The source audit checks original configs, metadata, model, data, snapshots and
adapter/token tensor equality. Existing results and checkpoints remain read-only.
Actual11/9/16 training images and captions are copied into review pages.

Use a fresh committed checkout and new external experiment directory. GPU0
must be idle both before source audit and immediately before model loading.
The methods run serially. SDXL's GPUs1/2/3 and checkouts remain reserved.

```bash
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -B -u \
  -m experiments.sd21_mailbox_sct_v1.run --output "$NEW_ROOT/experiment"
```

Each generation ledger records all three loaded branches. token_gates.json
records the actual masks. Completion requires22 images and unchanged source
hashes/checkout; it is not a quality or disentanglement claim.
