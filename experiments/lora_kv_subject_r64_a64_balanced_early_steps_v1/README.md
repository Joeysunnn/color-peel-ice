# Balanced-aligned Subject LoRA early steps

This study repeats only the selected `balanced_aligned` Subject run from
`lora_kv_subject_r64_a64_lr_step_ablation_v1`: `token_local_kv`, rank 64,
alpha 64, token-embedding learning rate `1e-5`, K/V learning rate `5e-5`, and
seed 42. The reviewed ten-row training data and all optimizer settings remain
unchanged.

It trains one continuous 700-step trajectory and saves exactly steps 300, 500,
and 700. After training, every snapshot is evaluated with Subject only,
literal metal, the selected `metal_spoon` and `wood_spoon` Material LoRAs, and
the fixed historical Subject transfer benchmark.
