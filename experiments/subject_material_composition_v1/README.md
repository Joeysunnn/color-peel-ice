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
repaired object masks. `protocols/mailbox_matte_counterfactual_v1.json` fixes a
masked Lab-lightness highlight suppression; the original images and every
background pixel remain unchanged. The preview shows each original beside its
derived matte candidate. Review must confirm weaker metal highlights while
preserving the mailbox shape, seams, decorations, color, and mask boundary.

After a matching preview review, staging creates a five-image matte-only S set
with the original captions and a ten-image metal/matte S set with explicit
ordinary material words. The intended training contrast keeps the earlier
token-local K/V recipe and 5,000-step total dose. Ten rows give each source
image fewer exposures than the five-row arm; interpret that contrast as a
balanced-data pilot, not an isolated material-word effect.
