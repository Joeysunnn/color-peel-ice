# SuperMat full-map Material LoRA

This study trains one independent rank-4, alpha-4 token-local K/V Material LoRA
for each of the three reviewed SuperMat counterfactual datasets: painted
mailbox, metal spoon, and wood spoon. The 72 training images, object masks,
captions, SD 1.4 base, optimizer, seed 42, and 5,000-step schedule come from
`natural_material_fullmaps_v1`. The adaptation changes from the original
token-local K/V weights to `material_lora_mode=token_local_kv`. The token
initializer changes from `material` to `metal`, as required by the existing
Material LoRA training entrypoint. This affects the initial token embedding,
so the two studies are not a single-variable adapter ablation. The three
2026-10-04 full-map checkpoints are **not LoRA checkpoints** and must not be
reported as this study's result.

The training configs are tracked in `configs/`. Before each run,
`verify_training_source.py` checks the completed source run, source config,
72 staged image/mask hashes, and that the new config changes only the adapter
and required initializer.
Save its JSON output with the new run artifacts. Launch from a clean server
checkout updated through GitHub fetch/fast-forward as required by
`doc/project-layout.md`. Use a fresh immutable run directory for every dry run
and actual run. The configs target GPUs 0, 2, and 1; confirm each is free
before starting concurrent jobs.

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
conda activate colorpeel017
material=metal_spoon  # repeat for mailbox and wood_spoon
config=experiments/natural_material_fullmaps_lora_v1/configs/$material.json
python experiments/natural_material_fullmaps_lora_v1/verify_training_source.py \
  --config "$config" \
  --output "$COLORPEEL_RUN_ROOT/natural_material_fullmaps_lora_v1/preflight_${material}_$(git rev-parse --short=7 HEAD).json"
run_id="$(date -u +%Y%m%d-%H%M%S)__natural_material_fullmaps_lora_v1__${material}_fullmaps_token_local_kv_lora_r4_5000__$(git rev-parse --short=7 HEAD)__42"
python scripts/launch/colorpeel_run.py --config "$config" \
  --run-dir "$COLORPEEL_RUN_ROOT/natural_material_fullmaps_lora_v1/$run_id" --dry-run
# After reviewing the dry run, create a different run_id and omit --dry-run.
```

Transfer uses the same five objects, seeds 42–44, 100-step PNDM sampling, and
CFG 3.5 as the existing LoRA study. For each material, `evaluate_transfer.py`
generates 45 images across base, literal, and `<M*>` arms, and writes prompts,
checkpoint hashes, safety flags, and contact sheets. The literal phrases are
the ones already used by `natural_material_fullmaps_v1`.

```bash
python experiments/natural_material_fullmaps_lora_v1/evaluate_transfer.py \
  --protocol experiments/natural_material_fullmaps_lora_v1/transfer_protocol.json \
  --material-id "$material" \
  --checkpoint "$COLORPEEL_RUN_ROOT/natural_material_fullmaps_lora_v1/$run_id/checkpoints" \
  --output "$COLORPEEL_RUN_ROOT/natural_material_fullmaps_lora_v1/transfer/$material" \
  --device cuda:0
```

The run ID above must refer to a completed training run; `--device` is relative
to the transfer process's `CUDA_VISIBLE_DEVICES` setting. Training or transfer
success is established only by the actual run manifests and image ledger.
