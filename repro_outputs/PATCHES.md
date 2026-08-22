# Patch record

## Verified implementation commit

- Branch: `repro/2026-08-21-colorpeel-clevr`
- Baseline: `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- Commit: `41d752a9d8e8b3a5ab711db90990ab28e4f58000`
- Message: `repro: add ColorPeel CLEVR 3x3 workflow`
- Highest verified risk: final modifier-token gradient-boundary correction
- README fidelity: adapted dataset/protocol; core ColorPeel training settings
  preserved; no training result claimed.

## Patch groups

### Data and study definition

- `src/methods/colorpeel_ice/prepare_clevr_3x3.py`
- `experiments/clevr_subject_color_3x3/`
- `tests/methods/colorpeel_ice/test_prepare_clevr_3x3.py`

Locks the nine IDs, 3×3 token map, RGB values, metal material, image-only
staging, hashes, and GT-mask audit boundary. Synthetic local tests passed; the
tracked script has not yet rerun against the real server root.

### Training correction and launcher

- `src/train/train_colorpeel.py`
- `src/train/token_gradient_utils.py`
- `scripts/methods/colorpeel_ice/train.sh`
- `scripts/launch/colorpeel_run.py`
- `tests/methods/colorpeel_ice/test_modifier_gradient_utils.py`
- `tests/test_experiment_runner.py`

The official loop omitted the last modifier ID when building its gradient
mask. The helper now preserves all six modifier rows and zeros all other
gradient rows. Local boundary and launcher tests passed. Literal official
AdamW decay of ordinary vocabulary values is intentionally allowed and must be
observed.

### Generation and evaluation adapters

- `scripts/methods/colorpeel_ice/generate.py`
- `scripts/methods/colorpeel_ice/score_clevr_predictions.py`
- `scripts/methods/colorpeel_ice/evaluate_color_metrics.py`
- corresponding tests under `tests/methods/colorpeel_ice/`

Loads all six tokens and Custom Diffusion weights, passes steps/CFG/generator
seeds, builds the 900-row manifest, emits CLEVR accuracy/contingency tables,
and computes color metrics from external masks. Local tests passed; real model
loading, generation, Grounded-SAM, Qwen3-VL, and metrics remain not run.

### Current pre-run handoff additions

- `experiments/clevr_subject_color_3x3/configs/smoke_2step.yaml`
- `experiments/clevr_subject_color_3x3/configs/smoke_9step.yaml`
- `scripts/methods/colorpeel_ice/segment_grounded_sam.py`
- `scripts/methods/colorpeel_ice/predict_qwen.py`
- tracked generation, segmentation, Qwen prediction, and scorer configs
- training observation fields and audit/report template updates

These pre-run handoff additions define the two real smoke contracts, literal
official AdamW drift policy, and independent evaluation stages. They are not
part of the currently server-verified commit `e6c57d1`; the server has not yet
been fast-forwarded to them. The isolated pytest suite passed 44 tests; this
does not turn any server stage into a completed run.

### Workflow and evidence structure

- `doc/`, `literature/`, `environment/`, and root `README.md`

Adapts the prior project's structural conventions only: method/study split,
tracked definitions, evidence labels, GitHub-only deployment, and external run
artifacts. No ICE method implementation was copied.

The official `src/test/test.py` behavior was restored during method-directory
reorganization. Its only remaining byte-level difference is a final newline
added by `apply_patch`; no executable line changed.

## Verification

```text
server GitHub checkout: clean at e6c57d1
earlier implementation snapshot: local checks completed
current pre-run handoff changes: isolated pytest 44 passed
training/generation/evaluation: not_run
```

## Explicit no-patch boundary

Do not add non-modifier embedding restoration. The tracked policy is
`literal_official_adamw_decay_allowed`: record ordinary-vocabulary drift, do
not restore it, and do not classify nonzero drift as failure. The two real
training smokes and the independent Grounded-SAM/Qwen3-VL stages remain
`not_run`; their output requirements are evidence contracts, not success
claims.
