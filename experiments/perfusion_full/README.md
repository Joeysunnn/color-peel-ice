# Full Perfusion on ColorPeel × ICE

This experiment applies the mechanism in [Perfusion (arXiv:2305.01644v2)](https://arxiv.org/abs/2305.01644) to both `<S*>` and `<M*>`. It is separate from `experiments/perfusion_subject_pilot` and all token-local K/V runs. The subject has two separately trained cohorts: original 25-image exposure-matched mailbox data and balanced aligned 10-image mailbox data. Material uses the selected 72-image ground-reflection metal source. The Material concept is an extension of the paper to a standalone material token; the paper's examples are object concepts.

## Method

- Every cross-attention layer applies the paper's covariance-metric Eq. (3) at **all text positions**. For each concept, the Key target is the frozen base projection of the contextual superclass token (`mailbox` or `metal`), and only its Value target plus the modifier token embedding are optimized. There is no learned free Key projection or token-local residual.
- The concept direction starts at the contextual superclass encoding and uses the paper's 0.99/0.01 EMA of the contextual modifier encoding during training. The Key target stays fixed.
- Two-concept inference applies the joint metric-span projection and both gates, algebraically equivalent to Appendix B's Cholesky/QR construction (see `test_attention.py`). Subject-only and Material-only inference load one concept each; the joint model is used only for prompts containing both.
- The covariance is uncentered and calculated from 100,000 distinct captions sampled from the official LAION-400M shard 0. One deterministic non-special contextual text token per caption contributes to the second moment. The paper does not specify a token-pooling rule; this choice and caption/sample hashes are recorded in the server covariance provenance.
- CLIPSeg soft object masks weight the denoising MSE. Each mask is normalized by its own maximum; no hard threshold is applied. The paper does not specify the exact weighted-loss reduction, so `train.py` uses a weighted spatial mean per image.
- Training uses SD 1.4, 400 optimizer steps, effective batch 16 via microbatch 1 and gradient accumulation, beta 0.75, tau 0.1, Value LR 0.03, embedding LR 0.006, seed 42, and checkpoints every 25 steps. The 400-step schedule, learning rates, and validation interval follow the paper. The asymmetric mailbox is not flipped; only cube/sphere material inputs may be flipped.
- Eight fixed prompts at each 25-step checkpoint are sampled with DDIM 50 and CFG 6 and scored by the harmonic mean of CLIP-I and CLIP-T. Scoring uses OpenAI CLIP ViT-B/32. The paper does not specify the exact validation prompt list, seed assignment, CLIP backbone, or reference aggregation, so these choices are explicit in `validate.py` and its selection artifact. Safety-filtered images are recorded and a checkpoint is eligible only if all eight are visible.
- The project comparison uses its frozen red/blue mailbox prompts, seeds 42–44, 100 steps, CFG 3.5, and the base SD pipeline scheduler. It also adds an uncolored control. This is a project comparison, separate from paper-style checkpoint validation. Both subject cohorts share the same newly trained Material concept.

The full method changes training duration, soft masks, covariance, and attention behavior relative to the earlier simplified P1. Any difference from that pilot or token-local K/V baseline is therefore descriptive, not a one-variable causal attribution.

## Files

- `fetch_laion_captions.py`, `covariance.py`: caption sampling and covariance/inverse construction.
- `prepare_soft_masks.py`: CLIPSeg object-mask preparation and image/source audit.
- `attention.py`: Eq. (3)/(4) projections.
- `train.py`, `configs/`: independent Subject and Material training.
- `inference.py`, `validate.py`, `compare.py`: paper-style selection and project-matched generation.
- `test_*.py`: mechanism, covariance, and mask checks.

Server artifacts live under `/home/r12user5/Documents/Jiawei/colorpeel-runs/perfusion_full/`. Existing checkpoints and outputs are read-only inputs.
