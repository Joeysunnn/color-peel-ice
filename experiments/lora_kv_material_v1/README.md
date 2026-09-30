# Material token-local K/V LoRA

This independent run trains `<M*>` using the same rank-4, alpha-4 token-local K/V LoRA mechanism as the balanced-aligned `<S*>` run. It uses the selected ground-reflection 72-image Material staging, the existing caption `a photo of an object made of <M*>`, seed 42, and the original 5,000-step training settings. No Subject or prior Material checkpoint is changed.

Launch `configs/ground_reflection_token_local_kv_r4_5000.yaml` with `scripts/launch/colorpeel_run.py` into a fresh directory below `$COLORPEEL_RUN_ROOT/lora_kv_material_v1/`. The launcher verifies the selected Material data, masks, preview authorization, hashes, and exact training settings before creating the run. It also requires a clean Git checkout and a run name of `TIMESTAMP__lora_kv_material_v1__ground_reflection_token_local_kv_r4_5000__COMMIT7__42`.

The checkpoint consists of `checkpoints/pytorch_lora_kv_weights.bin`, `checkpoints/<M*>.bin`, and `checkpoints/adaptation_config.json`. The adaptation config records `lora_material_kv`, `token_local_kv`, rank 4, alpha 4, and the modifier token. The LoRA state uses only cross-attention `to_k_lora` and `to_v_lora` tensors. Base SD weights and ordinary token embeddings remain frozen.
