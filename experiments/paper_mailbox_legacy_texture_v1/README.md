# Paper Mailbox Texture: accepted early data

The user confirmed reuse of the 20261018 Mailbox reviewed surface ROI,
q02/q98 grayscale extraction and actual T16 RGB/mask/caption files. These bytes
are validated by the existing natural-dataset verifier and copied unchanged by
the existing trainer. No new texture extraction, SuperMat run or rendering occurs.
The original reviewed mask is recorded as a material-surface mask, not relabelled
as the later SAM object mask. No old learned adapter or embedding is reused.

For subsequent paper Texture work, keep this early method: reviewed surface ROI,
existing robust grayscale extraction, then the established T16 renderer/trainer.
Do not substitute the automatic whole-object eroded ROI or the later gain1 method.
For Mailbox, use the frozen existing patch and T16 directly.

The newest selected-v1 Qwen VLM sees the original RGB crop at the frozen early
ROI [335,399,387,412]. This is only its observation input; it does not modify
the texture or training images. Its actual single-token initializer is retained.

Use a clean pushed checkout and the existing offline environments. Both stages
write only inside the supplied paper object root. First run:

```sh
python -u -B -m experiments.paper_mailbox_legacy_texture_v1.run \
  --stage prepare --object-root /absolute/paper/image_081__mask_130 --gpu 0
```

Then detach this finite training-to-inference queue:

```sh
python -u -B -m experiments.paper_mailbox_legacy_texture_v1.run \
  --stage train-and-infer --object-root /absolute/paper/image_081__mask_130 --gpu 0
```

The established SD2.1 rank64 token-local K/V recipe trains from base for500 steps
with paired300/500 exports. The existing paper transfer engine then generates
four exact chair/mug/vase/sofa Texture prompts, each with seeds20–60, at both
checkpoints:164+164=328 images. Sampling stays PNDM100/CFG3.5/512/FP16, checker on.

Outputs use a new `20261107__texture_legacy_t16__GIT7__42` tag under prepared/,
training/, ckpts/texture/ and provenance/. Inference uses sibling per-step
directories under infer/. Existing results are never deleted by this runner.
User-authorized removal of the specifically identified faulty prior Texture
weights and inference is a separate, path-audited operation.

Perform bounded startup verification only; do not monitor to completion.
