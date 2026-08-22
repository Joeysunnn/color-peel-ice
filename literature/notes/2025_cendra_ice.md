# ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models

- Authors: Fernando Julio Cendra, Kai Han
- Source: [arXiv:2503.19902](https://arxiv.org/abs/2503.19902)
- Venue: CVPR 2025
- Status: read
- Related study: [`clevr_subject_color_3x3`](../../experiments/clevr_subject_color_3x3/README.md)

## Paper evidence

- ICE addresses ambiguity in extracting interpretable visual concepts from a single image with a diffusion model.
- The paper describes a first stage for concept localization with text-based concepts and masks, followed by decomposition into intrinsic and general concepts.
- The authors report results for unsupervised intrinsic concept extraction.

This note records problem context only. It is not an instruction to transplant ICE stages or losses into ColorPeel.

## Our inference

- Subject/color entanglement is a relevant diagnostic context for the CLEVR study.
- ICE results do not establish that ColorPeel solves this problem, and ICE's masks or decomposition procedure would change the method under test.
- GT masks can support dataset audit or evaluation without entering ColorPeel training.

## Action

- Keep ICE method code, stages, losses, token parameterizations, and training protocol out of `colorpeel_ice`.
- Use ICE only to motivate bidirectional subject/color leakage reporting.
- Keep GT masks outside the training loss and disclose any external evaluation use.
- Track the actual ColorPeel-only evidence in [`01_colorpeel_clevr_baseline.md`](../../experiments/clevr_subject_color_3x3/reports/01_colorpeel_clevr_baseline.md).

## Evidence boundary

ICE is external context. No ICE baseline was run. The completed repository
experiment reports ColorPeel-on-CLEVR evidence only and does not treat ICE
metrics or training code as part of this method.
