# Five-matte Subject LoRA early-step trajectory

This experiment repeats `lora_kv_subject_matte5_r64_v1` without changing its
five training images, corrected captions, masks, optimizer settings, seed,
rank/alpha, adapter mode, or learning rates. It runs one continuous 700-step
trajectory and saves only steps 300, 500, and 700.

The training data remain read-only at
`${COLORPEEL_RUN_ROOT}/lora_kv_subject_matte5_r64_v1/assets_v1/staging`.
In particular, `red_matte.png` remains captioned as red and
`magenta_matte.png` remains captioned as purple.

The later inference phase must repeat the previous fixed matrix for all three
Subject snapshots: 60 comparison images and 140 historical transfer images per
step, for 600 images total. The Material-token arms remain the rank-64,
alpha-64, K/V-LR-`5e-5`, step-1000 `metal_spoon` and `wood_spoon` checkpoints.
