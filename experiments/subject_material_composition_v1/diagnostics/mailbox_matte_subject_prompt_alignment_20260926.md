# Mailbox matte S prompt-alignment diagnosis (2026-09-26)

## Frozen evaluation

- Protocol: `protocols/mailbox_matte_subject_inference_v1.json`; code commit `711b063`.
- Server output: `/home/r12user5/Documents/Jiawei/colorpeel-runs/subject_material_composition_v1/20260926-215429__subject_material_composition_v1__mailbox_matte_subject_inference_108__711b063__42/output`.
- All 108 samples finished. Baseline has 34 visible / 2 safety-filtered; matte-only and balanced each have 36 visible / 0 filtered.
- Red/blue S-only outputs at the same seed differ by mean absolute RGB 46.7/80.5/87.3 for baseline, 13.6/20.8/30.3 for matte-only, and 15.0/32.0/41.5 for balanced (seeds 42/43/44). This is an image-change diagnostic, not a color-accuracy metric. The new S arms are much less responsive to changing the color word.
- Visual review: at seed 42 both new S arms produce cyan mailboxes for red and blue prompts and repeatedly produce a wood-wall scene, even for the city-street prompt. The selected M can change highlights in some images, but cannot reliably override the learned S appearance.

## Inference mechanism audit

- The inference script loads each S checkpoint with `adaptation_mode="token_local_kv"`, installs the selected M as a second token-local residual, loads `<S*>` and `<M*>` embeddings, and passes separate S/M position masks under classifier-free guidance.
- The actual two-arm diagnostic found 16 dual token-local cross-attention processors for each arm. In S-only prompts, exactly one conditional S position (index 4) was marked and no M position was marked. The processor applies S and M residuals only at their respective positions.
- Thus the prompt-alignment failure is not evidence that full-position K/V was accidentally used. A strong residual at S can still change attention probabilities over all text tokens and carry background, color, layout, and material information in the S value.

## Training-data and residual checks

- All 10 balanced training rows use an identical object mask and exactly identical background pixels. They contain one mailbox pose/scene with five color labels and two finishes. Matte-only uses the same five matte rows. Changing finish did not provide independent subject views or contexts.
- Foreground median HSV hue in the staged data is approximately 332° for the row labeled `red` and 194° for the row labeled `blue`, for both finishes. These read visually as pink/magenta and cyan-blue rather than unambiguous red and blue. The color-label mismatch already exists in the old metal recolors; the new matte edit retained it.
- Last-100-step mean reconstruction loss: baseline 0.493, matte-only 0.155, balanced 0.263. These losses are not directly comparable across different image textures, but the much lower new-arm losses are consistent with easier memorization.
- Mean absolute learned S residual weight over 16 cross-attention processors: baseline K/V 0.002292/0.001996; matte-only 0.002583/0.002220; balanced 0.002547/0.002269. These values alone do not prove causality.
- Inference diagnostic `20260926-224955__subject_kv_strength_diagnostic__fb824fe__42` completed 16/16 images. Scaling both S K/V residuals to 0 or 0.5 restores visible red/blue prompt influence but loses the particular mailbox identity. Full scale restores identity and suppresses prompt influence.
- Split diagnostic `20260926-225722__subject_split_kv_diagnostic__a6b64f2__42` completed 12/12 images. Tested K/V scales 0/1, 0.5/1, and 1/0.5. None preserved the particular mailbox while reliably following red/blue prompts. This rules out these simple inference-scale settings as a satisfactory fix.

## Decision

Do not select matte-only or balanced S as the S+M subject checkpoint. Keep the prior S checkpoint as the current comparison baseline, while recognizing that it still carries metal appearance.

The next S-data revision should first align captions with measured and human-judged colors, then add independent backgrounds and, if available, different views of the same mailbox under each finish. Evaluate identity, color-word adherence, scene adherence, and M response at multiple training checkpoints before choosing a step count. The current one-photo matte/metal pairs cannot by themselves identify subject independently from finish, color, and scene.
