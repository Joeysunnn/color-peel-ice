# Mailbox Subject LoRA K/V ablation

This experiment trains four independent `<S*>` adapters from the same SD 1.4 base in each of two existing Subject data cohorts. It does not change the existing Subject or Material checkpoints.

| Mode | Key | Value |
| --- | --- | --- |
| `full_kv` | base + rank-4 LoRA at every text position | base + rank-4 LoRA at every text position |
| `token_local_kv` | base + rank-4 LoRA only at `<S*>` | base + rank-4 LoRA only at `<S*>` |
| `full_v` | frozen base (`ΔK=0`) | base + rank-4 LoRA at every text position |
| `token_local_v` | frozen base (`ΔK=0`) | base + rank-4 LoRA only at `<S*>` |

Only cross-attention K/V can change. Q, output projections, self-attention, the base SD weights, and ordinary text embeddings are frozen. The `<S*>` embedding follows the existing Subject training path. All arms use alpha 4, seed 42, 5000 steps, batch size 1, embedding and adapter learning rates `1e-5`, and no additional loss. The two data cohorts are `original` (25 exposure-matched source rows) and `balanced_aligned` (10 reviewed metal/matte rows). Their content hashes are checked by the run launcher.

Train with `scripts/launch/colorpeel_run.py` and one of the eight files in `configs/`. A run directory must be new and named `TIMESTAMP__lora_kv_subject_v1__VARIANT__COMMIT7__42` below `$COLORPEEL_RUN_ROOT/lora_kv_subject_v1/`. Each run saves its own `checkpoints/pytorch_lora_kv_weights.bin`, `<S*>.bin`, and `adaptation_config.json`.

After all four arms in a cohort succeed, run `evaluate.py` with its corresponding `protocols/comparison_*_v1.json` and the four `checkpoints/` directories. The evaluator checks each source run, loads the fixed selected token-local `<M*>`, and produces a matched 72-image comparison (4 modes × plain/red/blue × Subject-only/Subject+Material × seeds 42–44, 100 PNDM steps, CFG 3.5). Safety-filtered outputs are recorded separately in `generation_status.jsonl`. The full-position modes also change ordinary text positions and the unconditional CFG branch by design.
