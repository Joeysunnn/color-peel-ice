# Subject LoRA rank/alpha 64 K/V learning-rate ablation

This study keeps the reviewed balanced-aligned ten-row mailbox dataset, seed
42, token-embedding learning rate `1e-5`, optimizer, and four adapter scopes
fixed. It trains rank 64 / alpha 64 Subject LoRA with K/V learning rates
`1e-5` and `5e-5`.

Each of the eight runs follows one 3000-step trajectory and saves exact
snapshots at 600, 1000, 2000, and 3000 steps. GPU 1 hosts the full-position
arms; GPU 3 hosts the token-local arms.

The two locked protocols reuse the prior prompts, Material LoRA, seeds 42--46,
100 PNDM steps, and CFG 3.5. Each mode and snapshot compares Subject only,
Subject plus the literal word `metal`, and Subject plus `<M*>`.
