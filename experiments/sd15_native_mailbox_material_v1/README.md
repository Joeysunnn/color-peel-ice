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

## Wood wording comparison, seeds42-46

Add `--wood-wording` to generate15 images with three complete prompts:

- `a photo of a wooden mailbox`
- `a photo of a mailbox made entirely of wood`
- `a photo of a mailbox with a wood surface`

The photographic prefix is fixed to match the previous test. Columns follow that
order; rows are seeds42,43,44,45,46. All other weights, pipeline, precision,
sampler parameters, generator policy and safety handling remain unchanged.
Choose a new result directory; old results are not overwritten. The five-material,
ten-image mode remains the default when this flag is absent.

For the original surface wording at seeds42/43, `baseline_reproduction.json`
records whether newly generated PNG hashes equal the previous native10 results.
Hash disagreement is recorded rather than treated as a cross-environment guarantee.
