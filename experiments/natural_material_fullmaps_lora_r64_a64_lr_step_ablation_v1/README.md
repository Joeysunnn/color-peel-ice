# Natural full-map Material LoRA rank/LR ablation

This experiment trains only `token_local_kv` Material LoRA adapters on the three
reviewed natural full-map staging sets from `natural_material_fullmaps_lora_v1`:
`mailbox`, `metal_spoon`, and `wood_spoon`.

The matrix is rank 64, alpha 64, embedding LR `1e-5`, K/V LR in `{1e-5, 5e-5}`,
and one 3000-step trajectory per cell with exact checkpoints at 600, 1000, 2000,
and 3000 steps. Each source contains 72 staged renders (24 each for sphere, cube,
and cylinder). The launcher verifies the source manifests and every image/mask
hash before a run starts.

No file under either source study is modified. Run outputs belong directly under
`${COLORPEEL_RUN_ROOT}/natural_material_fullmaps_lora_r64_a64_lr_step_ablation_v1/`.

`protocols/inference_v1.json` locks the post-training evaluation. Each trajectory
generates 180 five-object transfer images (all four Material snapshots) and 180
new Subject/Material-token composition images (four Material snapshots by the
fixed Subject snapshots at 1000, 2000, and 3000). Existing Subject-only and
literal controls are intentionally not regenerated.

`protocols/subject_transfer_v1.json` separately locks the user's historical
140-row reconstruction/transfer manifest. `evaluate_subject_transfer.py` replays
those exact prompts and seeds for the selected 1000, 2000, or 3000-step Subject
snapshot without loading a Material adapter.
