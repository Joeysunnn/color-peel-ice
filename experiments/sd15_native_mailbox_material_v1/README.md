# Native SD1.5: mailbox material prompts

Generate10 images: the existing metallic, matte, glossy plastic, ceramic and wood
prompts, each with seeds42/43. Replace `<S*>` with the article `a`:
`a photo of a mailbox with a metallic surface`.

Use the same pinned SD1.5 generation weights as Subject experiments, stock
Diffusers StableDiffusionPipeline/UNet and its saved PNDM configuration. Keep
100steps,CFG3.5,512px,fp16 and a fresh CUDA generator per image. No learned token,
K/V update, LoRA, custom attention, ControlNet or image reference is loaded.
The project's shared SD1.4 safety checker remains enabled; this is the same
safety convention as previous comparisons, not the original SD1.5 checker.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export CUDA_VISIBLE_DEVICES=2
TPY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$TPY -B -u -m experiments.sd15_native_mailbox_material_v1.run \
  --output "$NEW_RUN" --device cuda:0 --execute
```

Omit `--execute` for a CPU-only source/hash/matrix preflight. Use a new external
output and a clean server checkout pulled from the local commit. Existing outputs
are rejected. Runtime records stock model/attention classes, absence of modifier
tokens, model hashes and actual scheduler configuration. Save all ten images,
prompts, seeds, safety flags, contact_sheet.png, index.html and repro_outputs/.

No training data are used. This tests the base model's material response; each
prompt can change generic mailbox shape and composition. It does not test
reconstruction of the photographed mailbox or establish disentanglement.
