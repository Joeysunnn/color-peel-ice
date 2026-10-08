# Subject LoRA alpha=8 step ablation

This training-only follow-up keeps the reviewed balanced-aligned ten-row
mailbox dataset, seed 42, rank 4, learning rates, optimizer, and four adapter
scopes fixed. It changes only Subject LoRA alpha from 4 to 8.

Each mode trains once to 3000 steps and saves nested snapshots at 1000, 2000,
and 3000 steps:

- `full_kv`
- `token_local_kv`
- `full_v`
- `token_local_v`

All four configs target GPU 0 so they can be run serially. Run directories must
be fresh direct children of
`$COLORPEEL_RUN_ROOT/lora_kv_subject_alpha8_step_ablation_v1/`.

The locked evaluation protocol in `protocols/comparison_v1.json` reuses the
alpha=4 study's prompts, Material LoRA, seeds 42--46, 100 PNDM steps, and CFG
3.5. For each adapter mode and Subject snapshot, it compares Subject only,
Subject plus the literal word `metal`, and Subject plus `<M*>`.

Run it with `experiments/lora_kv_subject_step_ablation_v1/evaluate.py`, passing
this study's protocol, matching alpha=8 Subject run, fresh output directory,
and CUDA device.
