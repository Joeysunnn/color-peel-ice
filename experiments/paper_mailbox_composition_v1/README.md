# Mailbox paper composition

User-selected S1000 / C150 / M150 / T300, 4 prompts × seeds20–60 = 164 images.
S/C use the frozen `cb169c6` exports; M uses the `16c1e7e` VLM training;
T uses the corrected early T16 workflow retrained at `ad3a393`.
The rejected structural Texture exports are never selected.

```text
a photo of <S*> in <C*> color with a <M*> surface with <T*> texture
a photo of <S*> in <C*> color with a <M*> surface with <T*> texture in the street
a photo of <S*> in <C*> color with a <M*> surface with <T*> texture in the snow
a photo of <S*> in <C*> color with a <M*> surface with <T*> texture surrounded by flowers
```

This runner reuses the existing `run_group(..., 'scmt', ...)` four-branch K/V
adapters and token gates unchanged. PNDM100, CFG3.5, 512px, FP16, empty negative,
checker enabled; all outputs and filter flags retained. Same-seed initial noise
is verified across prompts. No training, prompt rewriting or quality ranking.

```bash
python -u -B -m experiments.paper_mailbox_composition_v1.run \
  --object-root /home/r12user5/Documents/Jiawei/colorpeel-runs/paper_experiments_v1/image_081__mask_130 \
  --gpu 0
```

Launch from a clean committed checkout on an idle GPU. Source audit hashes,
paired exports and producer receipts are checked before loading. Color150 is
copied byte-for-byte to `ckpts/color/cb169c6/step150` under the existing object.
Results go to a fresh `infer/20261107__composition_S1000_C150_M150_T300__<git>__seeds20-60`
directory, with original PNGs, gallery, plan, generation ledger and source audit.
The Color copy destination must be absent on the first launch.
Only finite startup verification is requested; do not monitor to completion.
