# Texture–Material Disentangled Concept Learning v1

This independent study implements Phase 1 first. It reuses the existing
SD1.5 bundle, Blender 4.2.11 Cycles scene/lights/geometries/masks, shared
ColorPeel trainer, fullrank token-local K/V, rank64 alpha64 LoRA and dual
adapter inference. Existing Subject/Color experiments and results are inputs
only and remain unchanged. The previous SD1.4/SD1.5 comparison is a separate
study, not the scientific baseline for this new supervision protocol.

## Concept contract

- `<T*>`: spatial albedo pattern, including scale/direction/arrangement. Scheme A
  decodes sRGB albedo to linear Rec.709 luminance, normalizes valid-region
  2nd/98th percentiles to [0.25,0.75], clips outliers and rejects near-constant
  maps. It retains relative luminance, not source hue. Outside-mask pixels are
  neutral and excluded from quantiles. Albedo estimation errors can still leave
  shading/specular artifacts in Phase 2; grayscale is not intrinsic-image recovery.
- `<M*>`: scalar Roughness/Metallic only. No albedo texture, source color,
  normal/bump/displacement or source geometry enters the material representation.
- The renderer maps a Non-Color grayscale image through explicit UV coordinates,
  multiplies it by independently supplied linear RGB, and applies independent
  scalar R/M to Principled BSDF. UV scale/orientation stay fixed per texture.
  Existing asset UVs are retained; absent UVs get documented shape-specific
  projections. Seams and UV distortion are a first-version limitation.

## Phase 1 protocol

`protocol.json` freezes all factors and doses. Two target textures (vertical
stripes/checkerboard), two target materials (polished metal/rough dielectric),
and two auxiliary nuisance values of the opposite attribute give four datasets.
Each concept receives the same 2 geometries × 2 colors × 2 lights × 2 nuisance
conditions: 16 images. Total 64 training renders. T-training uses only auxiliary
smooth dielectric/rough metal; M-training uses only auxiliary horizontal
stripes/dots. Therefore all four target T×M appearance pairs are absent from
training, without correlating target identity with nuisance distribution.

Sixteen oracle reference renders cover those four target pairs under four
separate contexts: seen context/unseen pair, held-out cone, held-out green,
held-out top lighting. They are never staged into training. Paired target swaps
must preserve foreground masks, geometry, lights and seed. Blender uses 64
Cycles samples and the existing 512px lighting-v2 profile; this is a new pilot
profile, not a silent change to any old rendering experiment.

Each target concept gets an independent checkpoint containing the same generic
`<T*>` or `<M*>` token; checkpoint selection identifies the concept. Eight runs
cover 4 concepts × 2 methods. First perform two-step smokes, then exactly 300
steps per run. Embedding LR1e-5, original K/V LR1e-5, rank64 K/V LR5e-5,
alpha64, weight decay0, no CAA/attention/disentanglement loss, no hflip.
VAE, text transformer, base UNet and ordinary token embeddings stay frozen.
The shared trainer's only change admits `<T*>/texture` to the existing LoRA
token-local K/V allowlist. Its legacy metadata calls T `lora_subject_kv`;
the new training index explicitly records `branch: texture`.

Each completed run is checked for every metric step, finite adapter/token
tensors, expected shapes/rank, nonzero modifier update, zero ordinary embedding
drift and exact final snapshot/export equality. Smokes cannot be used as pilot
evaluation inputs. 300 steps tests execution and initial learning, not convergence.

## Reproducible execution

Develop locally, test, commit and push `codex/texture-material-v1`; on the
server fetch and pull `--ff-only`, then confirm the exact clean commit before
execution. Do not deploy code by SCP or edit it on the server. Use the existing
`colorpeel017` Python and the existing offline model/assets. Every output path
must be new and external; failed directories are preserved, not retried in place.

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
OUT=$COLORPEEL_OUTPUT_ROOT/texture_material_disentangled_v1/NEW_RUN
$PY -B -m experiments.texture_material_disentangled_v1.run_phase1 \
  --output "$OUT" --render-gpu 3 --train-gpus 1 2 3
```

The driver executes plan → 80 renders → verified staging → eight two-step
smokes → eight 300-step pilots → 256 evaluation images. It stops on a failed
stage. Per-stage commands, logs, hashes, status and scientific limitations go
to the new `repro_outputs/`. Individual stages are reproducible through:

```bash
$PY -B -m experiments.texture_material_disentangled_v1.data plan --output NEW_PLAN
# Blender CLI: see render.py and the driver's generated COMMANDS.md.
$PY -B -m experiments.texture_material_disentangled_v1.data stage --renders RENDERS --output NEW_STAGING
$PY -B -m experiments.texture_material_disentangled_v1.train --staging STAGING --output NEW_BATCH --mode smoke --gpus 1 2 3
$PY -B -m experiments.texture_material_disentangled_v1.train --staging STAGING --output NEW_BATCH --mode pilot --gpus 1 2 3
$PY -B -m experiments.texture_material_disentangled_v1.infer --training-index PILOT/training_index.json --output NEW_INFERENCE --device cuda:0
$PY -B -m experiments.texture_material_disentangled_v1.evaluate --inference INFERENCE --renders RENDERS --output NEW_EVALUATION
```

## Evaluation and claims

For both methods, 4 T×M pairs × 4 contexts × 4 conditions × 2 seeds produce
256 images. Conditions are literal T/M, learned T + literal M, literal T +
learned M, and learned T/M. Sampling is PNDM100, CFG3.5, 512px, fp16 and a fresh
CUDA generator per seed; the safety checker stays enabled. Both adapters are
installed per pair; absent tokens have exactly zero residual gates. T uses the
existing internal `subject` slot and M the `material` slot, without changing
the attention architecture or original S/C semantics.

Outputs include per-image ledgers, 16 paired contact sheets, HTML index,
blinded appearance descriptions CSV/HTML and an unblinding key. The separate
CPU evaluation stage audits complete image/condition hashes and adds 16
oracle-PBR-versus-generated comparison sheets with its own report. Assess
texture, reflective appearance, shape, color, light and background separately. Filtered
images are missing observations. No execution count, raw image-frequency
statistic or uncalibrated VLM label is presented as BRDF/disentanglement accuracy.
Existing material evaluator calibration found strong metal/rubber label bias;
see `clevr_subject_color_material_3x3x2/reports/08_material_evaluator_calibration.md`.
Counterfactual supervision is a hypothesis: prior R/M-only experiments still
showed shape leakage (`natural_material_threeway_v1/RESULTS.md`).

## Phase 2 and 3 gates

These phases are intentionally not executed by this first driver. Phase 1
appearance and leakage evidence must be reviewed first.

Phase 2 retains the existing SuperMat input/model/configuration and
`natural_material_supermat_v1/material_pipeline.py aggregate`: reviewed-region
8-bit R/M maps divided by255, median with MAD/IQR/quantile audit. Feed only its
Albedo to Scheme A (`data.extract_texture`), then use the same controlled
renderer. Do not reuse the old full A/R/M-map Material definition or introduce
IID. Compare against oracle source attributes and distinguish extraction errors
from learning/composition errors. Single-image BRDF is an estimate, not truth.

Phase 3 will reuse selected frozen S/C checkpoints with the new independently
validated T/M concepts, separately testing individual controls, object transfer,
unseen combinations and reconstruction. Four-token composition is not claimed
implemented by the current two-token pilot. No old S/C checkpoint is retrained.

## Additional training steps: 500 / 750 / 1000

`step_ablation.py` keeps the original protocol, staged image/mask files, eight
concept/method recipes, seed42, optimizer and learning rates unchanged. It
changes only the total dose to1000 and explicit saves to300/500/750/1000.
The shared trainer does not implement resume, so these are new runs from the
same initialization. Each new300-step adapter/token pair must exactly equal
its old300-step final export before inference proceeds.

```bash
$PY -B -m experiments.texture_material_disentangled_v1.step_ablation \
  --source-run "$ORIGINAL_PHASE1_RUN" --output "$NEW_STEP_RUN" \
  --gpus 1 2 3 --inference-gpu 3
```

Results are separated into `step500/`, `step750/` and `step1000/`, each with
paired snapshot exports, `inference/images/`, contact sheets, and
`evaluation/comparison_pages/`. Each step replays the original256 prompts/seeds
and PNDM100/CFG3.5/512/fp16/safety settings. Total768 new images. New step indices
record both the selected step and the source1000-step training run. The old
300-step results and original oracle renders remain read-only.
