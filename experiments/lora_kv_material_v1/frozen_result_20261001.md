# Selected LoRA Subject/Material result snapshot (2026-10-01)

The completed balanced-aligned token-local K/V Subject LoRA and ground-reflection token-local K/V Material LoRA were copied into a separate server snapshot before planning further material-learning experiments. The original runs remain at their recorded paths in `results_20260930.md` and `diagnostics_20260930.md`.

Snapshot directory:

`/home/r12user5/Documents/Jiawei/colorpeel-runs/lora_kv_frozen_20261001__selected_subject_material/`

The snapshot contains the final Subject and Material adapters and token embeddings, adaptation configs, embedding audits, training manifests/configs/metrics, the full 36-image matched inference result, all five completed diagnostic runs, and copies of the comparison protocol and reports. It does not duplicate the large intermediate training checkpoint directories or the other seven Subject LoRA arms; those remain in their original independent run directories.

`snapshot_manifest.json` records 249 copied files, their byte counts, SHA-256 hashes, source run paths, and code commit `8eb3fa9e55c6c9adde440c1a887262faca1b1e24`. Its own SHA-256 is `0e75b3ddb11e3f24990d56ea0aa8943f9070fdb5bf33337398c9c56c4612b0ea`. `checksums.sha256` passed `sha256sum -c --status` for all 249 files. The comparison copy contains 36 PNG images. The snapshot occupies approximately 395 MiB.

Key adapter hashes in the snapshot:

| Artifact | SHA-256 |
| --- | --- |
| Subject `pytorch_lora_kv_weights.bin` | `c89b994ee1988aaeef3c6842ba2c1aec4dbfd35b87874dcc33bacb6d94222acb` |
| Material `pytorch_lora_kv_weights.bin` | `9713ae80319aabb7ec08faa4349cfa2e03918bab915228bfa447afea6d7f170b` |
| Material `<M*>.bin` | `ef012da735605359db80033052fff3b06710b1bb081bb59782ac039ce192c7c6` |

This is the fixed reference for subsequent material extraction experiments; write all new code, checkpoints, and inference images to separate directories.
