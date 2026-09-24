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
