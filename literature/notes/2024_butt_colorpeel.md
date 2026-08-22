# ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement

- Authors: Muhammad Atif Butt, Kai Wang, Javier Vazquez-Corral, Joost van de Weijer
- Source: [arXiv:2407.07197](https://arxiv.org/abs/2407.07197)
- Venue: ECCV 2024
- Status: read
- Related study: [`clevr_subject_color_3x3`](../../experiments/clevr_subject_color_3x3/README.md)

## Paper evidence

- The paper frames named color prompt learning as learning user-selected, more precise color prompts than broad linguistic color names.
- The authors report that existing personalization approaches can entangle color and shape.
- ColorPeel uses multiple basic geometric objects in a target color to support color/shape disentanglement and reports color-generation experiments.

This section is a high-level evidence record, not a replacement implementation specification. Exact executable parameters for this repository remain in the tracked baseline config and must be checked against the pinned official source.

## Our inference

- A full subject/color Cartesian grid is a reasonable controlled setting for testing whether ColorPeel transfers to CLEVR.
- Sharing each subject token across three colors and each color token across three subjects may expose both leakage directions.
- The transfer is not paper-identical because the dataset and token allocation change from the official demonstration.
- Fixing the final-token gradient defect and resolving AdamW updates require separate comparability disclosure; neither follows merely from the paper's result claims.

## Action

- Lock the nine CLEVR samples and six tokens in the study manifest and baseline config.
- Preserve SD 1.4, Custom Diffusion K/V training, and CAA weight 0.2 from the official code path.
- Record official color metrics separately from CLEVR-specific accuracy and leakage tables.
- Resolve the AdamW non-modifier embedding conflict before training.
- Track all evidence in [`01_colorpeel_clevr_baseline.md`](../../experiments/clevr_subject_color_3x3/reports/01_colorpeel_clevr_baseline.md).

## Evidence boundary

The paper's results do not confirm this CLEVR experiment. Current repository findings remain **pending**.
