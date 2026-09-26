# Subject and material composition diagnostic

This stage tests the existing mailbox base-5 `<S*>` token-local K/V checkpoint
with the project owner's selected ground-reflection metal `<M*>` checkpoint.
Both adapters stay frozen and are applied only at their own token positions.
No new training images or joint checkpoint are created.

`protocols/mailbox_metal_diagnostic_v1.json` locks the two checkpoints and a
36-image grid: red, blue, and city-street prompts; subject-only, material-only,
subject plus material, and subject plus ordinary `metal` controls; seeds
42–44 at 100 steps and CFG 3.5. The safety checker remains enabled.

Compare mailbox identity cues, reflective metal appearance, requested color,
scene context, shape changes, extra objects, and filtering separately. This
is an inference diagnostic, not evidence that the factors are disentangled.

## Mailbox surface counterfactual preview

The next S-data ablation starts from the same five verified mailbox images and
repaired object masks. The v1 masked Lab-lightness preview still looked glossy
and was rejected before staging. `protocols/mailbox_matte_counterfactual_v2.json`
fixes a photorealistic matte reference edited from the green image, resizes it,
recolors it to the five source hues, and copies only original-mask pixels onto
each original background. The reference is a generated counterfactual, not
ground-truth relighting. The preview shows each original beside its derived
matte candidate. Review must confirm weaker metal highlights while preserving
the mailbox shape, seams, decorations, color, and mask boundary.

After a matching preview review, staging creates a five-image matte-only S set
with the original captions and a ten-image metal/matte S set with explicit
ordinary material words. The intended training contrast keeps the earlier
token-local K/V recipe and 5,000-step total dose. Ten rows give each source
image fewer exposures than the five-row arm; interpret that contrast as a
balanced-data pilot, not an isolated material-word effect.

The two 5,000-step token-local K/V recipes are staged in
`configs/mailbox_subject_matte_only_token_local_kv_5000.yaml` and
`configs/mailbox_subject_balanced_metal_matte_token_local_kv_5000.yaml`.
Both remain pending until the v2 preview is reviewed. After training, compare
the old S checkpoint with both new S checkpoints using fixed prompts, seeds,
and the same selected M checkpoint. Include S only, S with literal matte,
S with M, and S with literal metal conditions; score identity and surface
appearance separately. A matte source changes texture and may alter some
fine mailbox details, so the comparison tests whether material leakage is
reduced, not whether S and M are fully independent.

`protocols/mailbox_matte_subject_inference_v1.json` locks the completed old,
matte-only, and balanced S checkpoints plus the previously selected M. It
requests 108 matched images: three S arms, red/blue/city-street prompts, S only,
S with literal matte plastic, S with M, and S with literal metal, using seeds
42–44, 100 steps, and guidance 3.5. Compare shape/identity, requested color,
and apparent surface separately; filtered images remain in the status ledger.
