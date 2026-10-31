# Stock SD1.5 spoon CFG comparison

Generate exactly15 images: CFG3.5/5/7.5 × seeds42–46 inclusive, with fixed
`a photo of a spoon`, PNDM100,512px,fp16,enabled shared safety checker.
Each image resets a CUDA generator to its seed, so corresponding CFG columns
start from the same seeded noise. No negative prompt or extra conditioning.

Use **stock** StableDiffusionPipeline, UNet and attention from Diffusers0.17.
Do not import the project's custom UNet or load any learned token, K/V or LoRA.
Verify runtime pipeline/UNet/processor classes, vocabulary, scheduler and checker.
The frozen SD1.5 model and the project's existing checker bundle are unchanged.
Every image in this matrix is freshly generated through the stock entry point.
The old custom-entry spoon seed42/CFG3.5 PNG is only compared read-only, never reused.

```sh
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
CUDA_VISIBLE_DEVICES=2 /home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.sd15_native_spoon_cfg_v1.run --output "$NEW_EXTERNAL_ROOT" --execute
```

Keep all15 outputs in one fresh directory. `index.html` updates as images are
written; final `contact_sheet.png` shows five seed rows and three CFG columns.
Record exact prompts, PNG hashes, safety flags, stock runtime classes and model
hashes. Omit execute for a separate immutable preflight directory. Follow local
edit/test/commit/push then server clone/pull and execution. Old models/results
stay read-only. Verify the process and first PNG, then disconnect; no monitor,
automatic retry, training or checkpoint export.
