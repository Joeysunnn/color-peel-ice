# Subject LoRA step ablation

This experiment keeps the reviewed `caption_aligned_staged` ten-row mailbox
dataset fixed and changes only Subject LoRA scope and training dose. Four runs
train to 3000 steps with snapshots every 1000 steps:

- `full_kv`
- `token_local_kv`
- `full_v`
- `token_local_v`

The 1000/2000/3000 results are nested snapshots from one seeded trajectory per
mode, not independent replicates. The evaluator derives inference-only adapter
and `<S*>` files under its fresh output directory; it never writes back to a
training run.

Evaluation uses the completed Material LoRA checkpoint under
`$COLORPEEL_RUN_ROOT/lora_kv_material_v1/20260930-120700__lora_kv_material_v1__ground_reflection_token_local_kv_r4_5000__56dbb0e__42/checkpoints`.
For plain, red, and blue prompts and seeds 42--46, it compares Subject only,
Subject plus the ordinary word `metal`, and Subject plus `<M*>` at 100 PNDM
steps and CFG 3.5.

The `full_kv` and `full_v` Subject arms intentionally modify every text
position, including ordinary `metal`, `<M*>`, and the conditional/unconditional
CFG branches. Their three-condition differences are therefore not a pure test
of Material-token isolation. The two token-local arms provide that stricter
positional control.

Launch each file in `configs/` with `scripts/launch/colorpeel_run.py` into a new
directory below `$COLORPEEL_RUN_ROOT/lora_kv_subject_step_ablation_v1/`. Run
names must follow the standard `TIMESTAMP__STUDY__VARIANT__COMMIT7__42` form.
The four configs retain the existing GPU assignment 0--3.
