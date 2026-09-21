# Token-local K/V implementation audit

- Training enters the custom UNet through `src/train/train_colorpeel.py`; it tokenizes instance prompts into `batch["input_ids"]`, obtains text features, and calls `unet(noisy_latents, timesteps, encoder_hidden_states)`.
- Current `CustomDiffusionAttnProcessor` projects every cross-attention text position through `to_k_custom_diffusion` and `to_v_custom_diffusion`. Those layers are initialized from base K/V but replace, rather than add to, the base projections. Ordinary text tokens therefore receive learned K/V too.
- The custom UNet exposes `cross_attention_kwargs` and forwards them through cross-attention blocks. Its attention module forwards arbitrary kwargs to the processor. This provides a compatible explicit `[B, T]` modifier-token-mask channel.
- The selected implementation is a new processor mode: base `attn.to_k/to_v` stays frozen; zero-initialized learned delta projections are added only at discrete `<S*>` token positions. Self-attention and mask-free prompts use base K/V unchanged.
- Checkpoint tensor keys remain K/V-compatible, but token-local semantics require explicit metadata and a dedicated loader/inference path. Historical Custom Diffusion checkpoints remain interpreted by the existing loader.
