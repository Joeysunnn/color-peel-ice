# Material evaluator calibration

## Purpose

Human review found shape, color, metal and rubber stable and discernible, while
Qwen3-VL classified material correctly on only 1819/3240 generated images. The
error was asymmetric: all 1620 metal images were labeled metal, but only
199/1620 rubber images were labeled rubber. This stage tests whether that gap is
an evaluator limitation or non-canonical learned-rubber appearance.

## Locked protocol

- Run the identical three-key Qwen prompt on all 360 accepted v3 renderer
  references (`3 shapes x 3 colors x 2 materials x 20 views`).
- Keep Qwen3-VL-8B-Instruct, FP16, deterministic decoding, 128 output tokens,
  local-cache-only loading and GPU 3 unchanged.
- Report every prediction/failure and material confusion by shape/color.
- Build 162 blinded generated-image pairs: every fold, training seed, shape and
  color at generation seeds 42 and 43. Each pair differs only in material.
- Randomize review order and A/B side with seed 42; keep the key separate.

No numerical pass threshold is introduced. Human pair review remains primary.
Training, CAA, AdamW, token initializers and all checkpoints remain unchanged.

## Decision after review

- If Qwen also fails on accepted GT rubber, its generated-image material score
  is not a reliable absolute material metric; use the blinded pair result and
  document the evaluator limitation before considering two-object CLEVR.
- If Qwen recognizes GT rubber but not generated rubber, keep the experiment at
  single-object material and diagnose the learned rubber appearance before
  increasing object count.
