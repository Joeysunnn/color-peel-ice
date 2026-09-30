# Fixed-checkpoint Subject/Material diagnosis (2026-09-30)

This follows the completed Material LoRA comparison in `results_20260930.md`. The Subject is the frozen `balanced_aligned_token_local_kv_r4_5000` checkpoint; Material is the newly trained `ground_reflection_token_local_kv_r4_5000` checkpoint. The baseline uses the same plain/red/blue prompts, seeds 42–44, CFG 3.5, and 100 requested PNDM steps. Each diagnostic writes to a new run directory. No checkpoint or previous result was replaced.

## Stepwise controls

| Step | Single change or measurement | Completed run (under `/home/r12user5/Documents/Jiawei/colorpeel-runs/lora_kv_material_v1/`) | Observation |
| --- | --- | --- | --- |
| 1. Literal material word | Replace `<M*>` with ordinary `metal` in the nine S+M prompts. | `20260930-141100__literal_metal_diagnostic_9__b23934b__42` (9/9) | Literal `metal` did not reliably make the Subject metallic. New M with an ordinary mailbox sometimes did, especially plain seed 43. This points to a composition problem but does not isolate its cause. |
| 2. Subject checkpoint | S-only snapshots at steps 1000, 2000, 3000, 4000, and 5000. | `20260930-142100__subject_steps_s_only_45__5948ba4__42` (45/45) | Earlier snapshots sometimes remove duplicated parts, but also lose or alter the target mailbox identity. No tested step clearly solves shape stability. Step 5000 reproduced the original S-only images byte-for-byte. |
| 3. Material Value strength | S+M with M Value LoRA residual scale 0, 1, 1.5, or 2; M Key fixed at 1. | `20260930-143500__material_value_scale_36__ae2734f__42` (36/36) | Higher Value strength did not consistently add metal reflections; it sometimes shifted color, and shape failures remained. Scale 1 reproduced the original S+new-M images byte-for-byte. |
| 4. Attention measurement | Instrument the unchanged M-only and S+M inference. | `20260930-150000__material_attention_18__a167885__42` (18/18) | Across 9 paired prompt/seed cases, mean attention mass on `<M*>` fell from 0.007567 in M+ordinary-mailbox to 0.004877 in S+M, a 35.5% decrease. It was lower in all nine cases. Mean S-token mass in S+M was 0.115553. All instrumented images matched the original images by hash. Attention is descriptive, not a direct material-quality score. |
| 5. Material Key strength | S+M with M Key LoRA residual scale 0, 1, 1.5, or 2; M Value fixed at 1. | `20260930-151600__material_key_scale_36__204e680__42` (36/36) | Mean M-token attention rose with Key scale: 0.003075, 0.004877, 0.006934, 0.010700. Scale 1.5 nearly restored the M-only mean, but did not consistently strengthen visible metal. Scale 2 changed colors and sometimes composition/shape. Scale 1 reproduced the original S+new-M images byte-for-byte. |

All reported images completed with `ok` status and matched recorded image hashes. The successful attention run made 101 UNet calls per layer for 100 requested PNDM steps; an initial, incomplete attention run at `20260930-145100__material_attention_18__5505dae__42` stopped when the instrumentation incorrectly expected exactly 100 calls. The corrected run above is the one used for the measurements. This correction changed only the diagnostic accounting.

## Interpretation

The new Material adapter has a visible signal with an ordinary mailbox in some cases, while its effect is unreliable beside the learned Subject. `<M*>` receives less average cross-attention in S+M than in the M-only control. Restoring its attention mass with a stronger Key residual does **not** reliably restore metal appearance and can damage color/geometry. Increasing Value residual strength also fails the fixed-seed visual check. The evidence therefore does not support selecting a new K/V inference scale or another Subject training step from these sweeps.

The M-only control also changes the mailbox identity and prompt structure, so its attention difference is not a causal estimate of Subject suppression. The nine-image visual grid is qualitative and does not establish that every seed or prompt behaves this way. The stronger conclusion is limited to these fixed checkpoints and prompts: simple inference strength changes do not solve the observed S+M material weakness.

The next intervention should be an **independent Subject-data revision**, followed by the same fixed evaluation. An earlier dataset audit in `../subject_material_composition_v1/diagnostics/mailbox_matte_subject_prompt_alignment_20260926.md` found that all ten balanced Subject rows share one object mask and identical background pixels: one mailbox pose/scene, five color labels, two finishes. Those pairs do not provide independent views or contexts for learning identity apart from color, finish, and scene. Before retraining, collect or stage real independent views/backgrounds of the same mailbox with verified captions, then hold the Material checkpoint, inference protocol, and S method fixed. This is a data hypothesis to test, not a proven cause of the current failure.

## Review and provenance

Local side-by-side contact sheets are under `review/literal_metal_step1/contact_sheets/`, `review/subject_steps_step2/contact_sheets/`, `review/material_value_step3/contact_sheets/`, and `review/material_key_step5/contact_sheets/`; canonical full-resolution images and ledgers remain in the server runs above. The review copies are untracked; the server runs are the evidence source.

| Run step | `provenance.json` SHA-256 |
| --- | --- |
| Literal metal | `c0df51cbfdccefb6e9d2ac30dfaf9c7597f65c7e8d129567459f7f8365d3b325` |
| Subject snapshots | `dfe8cd50608bec739e641aba47f255dacb949c18a57e8e42925165774d98a131` |
| Material Value scales | `5b5dc3149c088a4c9a46e8a0a9c590695edd08b6294fc2f85a18330790a16bbe` |
| Attention measurement | `b17dbfa68993977081d5f91bc3c6815d403ffd78cc91aa2948847d9613a2a957` |
| Material Key scales | `6416bd3002412be2c995753c7b0eab834451bea1d5dd5c9b89ba4d5ae117bbf4` |
