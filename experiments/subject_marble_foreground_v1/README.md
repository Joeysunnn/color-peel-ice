# MARBLE: foreground-only material generation

This independent experiment removes the original scene before every image-conditioned
branch. It preserves the existing five RGB-LCh colors, Subject mask, official models,
prompts, Euler scheduler, seed42,30steps,CFG5,strength1,ControlNet1 and IPAdapter1.
It does not train a token or modify any previous experiment.

## Background paths and interventions

| Path | Previous five-color experiment | New intervention |
| --- | --- | --- |
| CLIP / IP-Adapter / both control MLPs | Entire1024 target was the reference | Explicit subject-only reference; masked before resizing; bbox+10percent margin; aspect-preserving neutral padding |
| DPT | Entire original scene | Fresh DPT from isolated target; output depth outside Subject mask set to0 |
| Initialization / VAE | Only subject converted to gray; original scene retained | Source background already replaced byRGB127 before official gray initialization |
| Inpaint context | Scene retained outside edit mask | Whole target uses neutral padding; mask is still the existing Subject region |
| Resizing | Scene could interpolate into boundary | Mask at512 before any resize; enforce neutral exterior again at1024 |
| Future35 training candidates | Original five recolors retained scene | All35 have neutral exterior; RGBA cutouts saved separately |

The existing mask excludes the upper blue/gold floral `135` sign. Original photographs
are copied for provenance and displayed in the input sheet; they are never passed to
CLIP, DPT, the MLPs or VAE. Original-scene depth and initialization caches are not reused.
Current official pipeline has a4-channel UNet; diffusers0.30 blends init-image latents
outside the edit mask every denoising step. Masking only the material reference did
not eliminate this source-scene path.

Target canvas, object location and scale remain fixed. The material reference is
separately cropped to increase foreground coverage in CLIP; thus this tests a joint
foreground/reference adaptation, not a pure background-only causal ablation.
Mask uses NEAREST rather than the old BICUBIC resize to avoid gray boundary masks.

Neutral padding still gives the model spatial context. Reflections, dirt, texture
already inside the object, mask errors and the generative prior remain. Removing
the scene does not guarantee identity, color, albedo or texture preservation.
RGB/highlight changes are not used as an albedo leakage verdict.

## Scope and execution

Generate five no-attribute diagnostics (empty MLP dict, still strength1 regeneration),
then the full five-color M[-20,+20] × R[-1,0,+1] matrix. Controls are relative learned
edit strengths, not physical PBR values. Each edit starts independently.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
MPY=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python
$MPY -B -m experiments.subject_marble_foreground_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$NEW_RUN" --gpu 2
```

Use a new clean checkout pulled from the local commit. Output must be new and external;
an existing directory is rejected. Source/model/code hashes are checked before and
after execution. Failed partial outputs remain preserved; select a new run if needed.
The generator has no training switch. Train only after the user's explicit approval.

## Artifacts

- `source/{color}/`: original photograph (provenance only), transparent cutout,
  neutral512 original, clean1024 target, cropped reference, fresh depth before/after
  exterior suppression, actual official gray initialization.
- `plan.json`, `source/conditions.json`, `runtime.json`: source and model-input hashes,
  reference crop boxes, exact sampling and environment.
- `raw/`: untouched35 model outputs (30 M/R +5 no-attribute diagnostics).
- `dataset/`:35 candidate images (30 generated +5 neutral original recolors), masks,
  proposed captions and one-image-per-directory concepts. No-attribute outputs excluded.
- `cutouts/`:35 RGBA candidates, transparent hidden RGB zeroed.
- `review/inputs.png`: original/provenance vs actual clean Target, cropped Reference, Mask.
- `review/overview35.png`, per-color old/new comparisons, no-attribute identity sheet,
  `review/index.html` and caption CSV: all visible source/result evidence.
- `repro_outputs/`: reproducibility notes and execution status.

Raw outputs remain untouched. Candidate postprocessing only replaces the generated
mask exterior with neutral padding; it never pastes original foreground RGB back.
Every candidate remains in pending-review state regardless of visual plausibility.

CPU checks: `python -B -m unittest discover -s tests -p 'test_marble_foreground*.py' -v`.
They include extreme-background invariance before resizing, explicit foreground-only
caller inputs, depth suppression and generated-foreground preservation in packaging.

## Training after user approval

The user subsequently authorized training on the completed foreground-only data.
Reuse the existing MARBLE35 training pipeline; staging recognizes this generation
schema, verifies all35 image/mask hashes and exact neutral exteriors, and copies
the approved candidates into a new independent training directory. The five
no-attribute diagnostic outputs are never included. The original pending-review
generation artifacts remain unchanged; approval is recorded in the new staging.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
TPY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$TPY -B -u -m experiments.subject_marble_color30_v1.train_pipeline \
  --candidates /var/tmp/colorpeel-r12user5/runs/subject_marble_foreground_v1/20261026-131331__foreground30_plus5__7bc1a86__42/dataset \
  --output "$NEW_TRAIN_RUN" --gpu 2 --execute
```

Original fullrank and rank64/alpha64 train serially from the same frozen SD1.5
base with the previous B/C/MARBLE35 recipes: S token, mailbox initializer,
seed42,1000 updates and500/750/1000 paired checkpoints. Data paths and study/status
metadata change; optimizer, learning rates, masks, precision, losses and sampling
remain unchanged. After both completed audits, infer the same five material
prompts with seeds42/43,PNDM100,CFG3.5,512px,fp16 and the existing safety checker.
Outputs use separate step500/750/1000 folders; every contact sheet shows all35
actual neutral-background training images and captions. Exact commands/configs,
source hashes, logs and status are written into the new run.

## Pink/cyan-only training ablation

The user's later two-color request selects exactly14 images from the same completed
foreground-only candidates: pink and cyan each contribute one neutral original and
all six M/R counterfactuals. Neither other colors nor no-attribute diagnostics are
copied into the training dataset. Source pixels, masks and captions remain byte-identical.
Use a new output and the same command above with `--pink-cyan-only --execute`.

Both methods restart from the frozen SD1.5 base with the previous initializer,
learning rates,seed42,1000 updates and500/750/1000 saves. The existing sequential
loader now cycles14 images:71 complete cycles plus the first6, giving pink502 and
cyan498 exposures. Each image receives71 or72 exposures; the increase from35-image
training is documented rather than changing the requested update count.
The new study is `subject_marble_foreground_pink_cyan_training_v1`.
Inference prompts,seeds42/43,PNDM100,CFG3.5,512px,fp16 and safety checker remain fixed;
each step's sheets show exactly the14 actual training images and their captions.
