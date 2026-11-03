# Single-image Qwen initializers on SD2.1

The first input is the original orange Mailbox photo, SHA256 `ba9c6624d9998cdfcd16308b0cf8fd8f3778909d8655efe2ba9cebba370483d0`. All four training datasets originate from that photo. Earlier wood-spoon Texture weights and the TokenVerse-inspired loss are excluded.

## Whole-method audit

| Stage | Verified current behavior | Integration or remaining limit |
| --- | --- | --- |
| Input/target | Original photo and reviewed foreground mask | Qwen reads the original RGB; existing multi-detection SAM command is not a general single-target intake |
| Subject | Recolor foreground; TRELLIS2 reconstruction; camera fit; 8 center-view counterfactual renders + 3 real-photo colors | Reuse approved11; original geometry defects remain, fixed view cannot reconstruct missing geometry |
| Color | Pixel-derived Lab target and bounded lightness; 9 shape/view renders | Qwen supplies only a color-name initializer, never replaces the measured target |
| Material | Same-source SuperMat estimate, reviewed ROI, scalar roughness/metallic; 8 counterfactual renders | Qwen separates possible substrate from visible finish; no physical M/R estimate is inferred from its words |
| Texture | Same-source grayscale albedo spatial patch; 16 nuisance-varied renders | No normal/displacement; no wood-grain substitution; a plain surface may have no useful distinct texture concept |
| Initialization | Prior Qwen scripts classified fixed CLEVR labels; trainer had historical initializer whitelist | New independent single-image schema, candidate ranking, actual SD2.1 single-BPE filter and hashed manifest; unknown stops training |
| Training | One learned token + token-local LoRA64 K/V per branch; frozen base and normal vocabulary | Opt-in audited manifest replaces only the whitelist; verifies actual copied1024-vector; decay0, auxiliary losses0, fresh base |
| Checkpoint | Adapter/token must come from the same saved step | Reuse strict SD2.1 snapshot audit and add initializer provenance |
| Inference | Existing single-branch and SCT loaders | New SCMT processor installs all four residuals once, disjoint text gates, negative-CFG gates0 |

The initializer must be one *actual diffusion tokenizer token*, not merely one English word. No first-BPE truncation or averaging. Caption lists must be nonempty, contain exactly their own learned modifier once, and fit77 tokens without truncation. Qwen candidates/descriptions and rejection reasons are preserved. M/T are separated by observable reflection versus spatial albedo pattern; ambiguous `smooth` is rejected for Texture.

Material captions change from `made of <M*>` to `with a <M*> surface` to fit an adjective such as glossy. Subject category words use the Qwen category. These changes are explicit: this is an integration test, **not a pure initializer-only causal ablation**. It does not by itself solve the earlier background-binding failure.

## Model and bounded protocol

Use [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), local cached revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, FP16/eager on V100, deterministic single-image generation. Record full weight hashes once and actual measured allocator memory. Weight file bytes are not a VRAM estimate. Use existing `ice-vlm` environment; leave `colorpeel017` untouched. [Official quickstart](https://github.com/QwenLM/Qwen3-VL#quickstart).

Four LoRA64/alpha64 branches, seed42, batch1, FP32, embedding LR1e-5, K/V LR5e-5. First2 updates per branch to validate startup and initialization, then fresh S/M/T1000 (save500/1000), C150 (save100/150). Existing per-branch augmentation and masked reconstruction recipe remain. No checkpoint resume or auxiliary preservation/isolation objective. Four currently idle GPUs are checked before use; never stop existing jobs.

`caption_token_ids` records the four-modifier preflight tokenizer only. Each training job adds one modifier; its actual ID and copied vector are in `initializer_audit.json`. The inherited Color `hflip=True` flag is retained, but the historical trainer does not execute that flip; this integration does not silently change augmentation behavior.

Inference:112 images, seeds31/38, PNDM100,512px, FP16.96 images cover each branch's native/literal/learned controls on4 cases;16 cover midpoint/final SCMT on plain/street/snow/flowers. Color-only CFG6; other branches and SCMT CFG3.5. Preserve same-seed initial latents, prompt token gates, exact loaded sources and safety-filter flags. Visual quality requires inspection after technical checks.

## Entry points

1. In `ice-vlm`: `python -B scripts/methods/colorpeel_ice/initialize_from_image.py --image ORIGINAL --model-dir PINNED_QWEN_SNAPSHOT --sd-model-dir PINNED_SD21 --output FRESH_VLM_DIR --device cuda:0`.
2. In `colorpeel017`: `python -B -m experiments.sd21_qwen_initializers_v1.run --stage prepare --initializer-manifest FRESH_VLM_DIR/initializers.json --output FRESH_EXPERIMENT`.
3. Same environment: `python -B -m experiments.sd21_qwen_initializers_v1.run --stage train --output FRESH_EXPERIMENT`.

Deploy tracked source only through reviewed commit/push and a fresh server clone. Store data, model receipts, logs and outputs outside Git. The semantic entry accepts other images, but this data resolver deliberately accepts only the audited Mailbox SHA. A new image still needs its target mask, TRELLIS/SuperMat processing and derived-data review before four-branch training; it must never silently inherit Mailbox assets.
