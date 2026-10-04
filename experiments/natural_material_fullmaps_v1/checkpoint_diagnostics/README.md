# Material checkpoint progression diagnostic

This run compares only the metal-spoon full-map `<M*>` checkpoint at steps
1000, 2000, 3000, 4000 and 5000. It uses two material-only prompts:
`a photo of a cone made of <M*>` and
`a photo of a mailbox made of <M*>`, seeds 42–44, SD 1.4 with PNDM,
100 sampling steps and CFG 3.5. It does not test the prompt without
`a photo of`; that comparison is a separate matched evaluation.

The final step is copied with verified hashes from the existing transfer run.
Earlier steps are derived from the unmodified Accelerate training snapshots:
`pytorch_model.bin` supplies token-local K/V weights and
`pytorch_model_1.bin` supplies the trained `<M*>` embedding row. The script
checks that the 5000-step snapshot exactly equals the exported final files
before generating any images. All source paths, hashes, prompts, images,
safety flags and contact sheets are recorded under a new output directory.

Example on research12 (with a new output directory):

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/natural_material_fullmaps_v1/checkpoint_diagnostics/evaluate_steps.py \
  --protocol experiments/natural_material_fullmaps_v1/checkpoint_diagnostics/protocol.json \
  --checkpoint-run /home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_fullmaps_v1/20261004-114311__natural_material_fullmaps_v1__metal_spoon_fullmaps_token_local_kv_5000__241c32a__42 \
  --final-transfer /home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_fullmaps_v1/run_20261004_fullmaps_threeway_lighting_v2/transfer/metal_spoon \
  --output /home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_fullmaps_v1/checkpoint_diagnostics_metal_spoon_20261004 \
  --device cuda:0
```

`cuda:0` refers to physical GPU 1 inside `CUDA_VISIBLE_DEVICES=1`.
