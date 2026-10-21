# MARBLE C first six-image pilot, 2026-10-21

Execution is complete. **The six-image pilot did not pass data-quality review;
all six are rejected for Subject training.** No30-image expansion, hue editing,
training concepts, Subject training or learned-token A/B/C comparison was run.
This is one mailbox/source/reference/control-grid observation, not a claim
that MARBLE fails for all inputs or configurations.

## Immutable execution and inputs

Project code: `86fb6a28aa9f2602515d16deb6cae063760dde27`, clean branch
`codex/subject-marble-c-v1`. Official MARBLE: 
`0139f2563057e8ac7d8e82055ea891ffd7474bd5`.

```text
Server run:
/var/tmp/colorpeel-r12user5/runs/subject_marble_c_v1/20261021-125714__phase1_six__86fb6a2__42

Isolated server checkout:
/var/tmp/colorpeel-r12user5/checkouts/subject-marble-c-v1

Pinned official code, model weights, independent environment and receipt:
/var/tmp/colorpeel-r12user5/assets/marble_c_v1
```

The source is the original orange512px mailbox JPEG, SHA256
`ba9c6624d9998cdfcd16308b0cf8fd8f3778909d8655efe2ba9cebba370483d0`.
The reused42951-pixel Subject mask, SHA256
`550c1e13bee1435e1145f06e44bdef02d75b8b8fdfa7462d3b39f456dfc9a978`,
contains the previously checked682-pixel bottom-cut completion. No PBR geometry,
Albedo, rendered subject or B's erroneous enlarged mask was used.

Every edit independently starts from the same original RGB and uses that
whole photograph as the official default texture reference. A single official
DPT prediction is shared across six calls. Metallic low/high=+20/-20;
Roughness low/medium/high=-1/0/+1. These are relative MLP editing strengths,
not absolute PBR M/R. No output is used as the next input.

Actual runtime: Tesla V100-SXM2-32GB, CUDA-visible GPU2,
Torch2.5.1+cu124, diffusers0.30.0, transformers4.27.4,
huggingface_hub0.30.2, accelerate0.26.1, NumPy1.26.4, rembg2.0.61.
The official `StableDiffusionXLControlNetInpaintPipeline` uses
EulerDiscreteScheduler, CFG5.0,30steps,seed42,strength1,ControlNet1,
IPAdapter1; generic official positive/negative prompts are unchanged.
Model CPU offload is enabled. The local-path and Torch-version adaptations
are declared in README/runtime.json; official editing code was not patched.

## Observed quality

| Relative Metallic | Relative Roughness | Hue median drift | Raw interior edge recall | Verdict |
|---|---|---:|---:|---|
| Low(+20) | Low(-1) | 14.12deg | 0.594 | Rejected: new lettering and ornament |
| Low(+20) | Medium(0) | 15.53deg | 0.592 | Rejected: new lettering and ornament |
| Low(+20) | High(+1) | 16.94deg | 0.620 | Rejected: local detail/color changes |
| High(-20) | Low(-1) | 29.65deg | 0.624 | Rejected: blue/gold hue and texture |
| High(-20) | Medium(0) | 31.06deg | 0.622 | Rejected: blue/gold hue and texture |
| High(-20) | High(+1) | 31.06deg | 0.633 | Rejected: blue/gold hue and texture |

Hue compares saturation-gated HSV pixels; reflective changes affect it.
Gradient edge recall uses2px tolerance and the raw image resized to512,
before copying background. These are screening proxies, not segmentation
IoU, measured color constancy or proof of geometric deformation. The final
verdict uses the full six-image grid plus native raw/final image inspection.

- High Metallic visibly changes reflection appearance, but the body becomes
  blue/gold and acquires new surface patterns. This violates the fixed-original-
  color and preserved-detail requirements even though it looks reflective.
- Low Metallic invents white/blue lettering, ornate backplate motifs and
  textures; mailbox slots/handle/part detail no longer match the source.
  Conditioning the S token on these images would supervise those unwanted
  properties along with identity.
- Increasing Roughness has a smoothing trend: foreground mean gradient
  falls11.42→9.86→8.72 in low-M and13.30→12.49→11.93 in high-M. However,
  texture/part detail also changes. The high-M reflection band remains strong
  in all three; this pilot does not establish clear independent three-level
  highlight-width control.
- Raw-background MAE, after raw1024→512 LANCZOS resizing, is4.013–4.046/255.
  It includes generative/VAE and resampling effects. Final candidates have
  exactly0 changed pixels outside the mask only because original background
  RGB is explicitly copied back. Raw1024 outputs remain separately available.
- Best global brightness scaling cannot explain M low/high differences
  (relative residual0.460–0.466), but those differences include color/texture
  leakage and cannot be interpreted as isolated metallic accuracy.

## Interpretation and next bounded test

The official function converts the editable foreground to grayscale and uses
strength1 inpainting. Depth and mask constrain layout, but do not hard-lock
original RGB, local texture or individual parts. MLP control edits a CLIP
reference representation rather than a recovered object BRDF. Thus a successful
material-looking edit is not automatically a valid subject-disentanglement
counterfactual.

One unverified contamination hypothesis is the whole-photo texture reference:
the prominent blue/gold ceramic number plaques/background can affect the
reference embedding. The edited blue/gold body and invented lettering are
consistent with this possibility, but this pilot did not perform the object-
only-reference ablation required to establish causality. A small test using
an explicitly documented Subject-region reference would be a next step before
any30-image expansion. No such alternative or lower-strength rewrite was
silently substituted in this run.

## Evidence, reproduction and preservation

The README provides exact setup/assets/run commands. `assets.json` pins HF
revisions and hashes every downloaded model and official tracked source file.
All locks and the official clean checkout were verified before/after inference.
Local and server phase tests each passed15 cases. The downloaded six raw1024
and six final512 PNGs all match their generation-manifest hashes.

The run contains `raw/`, `images/`, `source/`, `plan.json`, `runtime.json`,
`generation_manifest.jsonl`, `pairwise_response.json`, `contact_sheet.png`,
`index.html` and `repro_outputs/`. Its generation `status.json` records the
completed candidate stage; the appended `phase1_review.json` records the later
visual verdict with **six explicit rejected decisions**, zero eligible images,
and no expansion/training. Proposed captions are saved as hypotheses, not
verified material labels. No concepts file was created.

All project edits were local, committed and pushed, then pulled into the new
server checkout. Existing B, SuperMat/PBR C code branches, environments,
checkpoints and results were not overwritten or rerun. The ColorPeel CUDA
training environment was not modified; the MARBLE package overlay is isolated.
