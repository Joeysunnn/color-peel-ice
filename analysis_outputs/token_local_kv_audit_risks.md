# Token-local K/V audit risks

- Existing generic inference uses the installed Diffusers loader, which would interpret the new K/V tensors as ordinary Custom Diffusion projections. Token-local runs require a dedicated loader that constructs the new processor and passes the discrete mask through `cross_attention_kwargs`.
- Classifier-free guidance duplicates text conditioning. Inference must create an aligned mask for both unconditional and conditional text sequences.
- The project has an xFormers Custom Diffusion processor without a token-mask interface. The initial token-local implementation must reject xFormers rather than silently lose gating.
- Historical attention checkpoints do not contain adaptation-mode metadata. New runs need an explicit `adaptation_config.json`; no historical artifact should be rewritten.
