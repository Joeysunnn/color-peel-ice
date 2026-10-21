# MARBLE-based Subject C, phase1 only

Independent from `subject_albedo_pbr_c`: no SuperMat extraction, PBR geometry,
PBR image reuse, new losses or Subject training. Current scope is six original-
color material-edit candidates and quality review. There is no30-image or
training entrypoint. Later training methods are outside this phase.

The first executed six-image grid and its **failed quality verdict** are
recorded in [PHASE1_RESULTS.md](PHASE1_RESULTS.md). Generation completed;
none of those images is accepted for training.

## Official implementation

Source: https://github.com/Stability-AI/marble at
`0139f2563057e8ac7d8e82055ea891ffd7474bd5`. The official
`parametric_control.ipynb` and `marble.run_parametric_control` are authoritative.
Metallic low/high use **+20/-20**, while Roughness low/medium/high use-1/0/+1.
These are relative learned controls, not physical M=0/1 or measured roughness.
Even control0 need not imply an identity edit.

The original512px orange mailbox photograph is frozen by SHA256. The existing
42951-pixel mask includes the previously reviewed682-pixel bottom-cut repair;
only this mask is reused from the previous C. The earlier B enlarged mask,
which contained background-post additions, is not used. Each MARBLE call starts
from the same original image, with the same original image as the default
texture reference; the six outputs are never chained. No metal-bowl reference,
prompt-based material substitution or imagegen is used.

DPT depth is computed once with official preprocessing and shared across all
six calls. The official RGB mask path resizes to1024 using its existing default.
Seed42,30steps,strength1,ControlNet1,IPAdapter1 and generic positive/negative
prompts stay at official defaults. CFG is the unchanged SDXL ControlNetInpaint
pipeline default5.0, not the unrelated IPAdapter generate method's7.5. Actual
pipeline/scheduler/config and package versions are saved.

## Isolated setup and execution

All project changes are made locally and pushed; the server uses a new clean
checkout of `codex/subject-marble-c-v1`. Old C and B checkouts stay untouched.
An isolated Python3.11 virtualenv may inherit the existing CUDA-compatible
Torch2.5.1+cu124 from `flux-smoke`; its local package pins in requirements.txt
restore official non-GUI inference versions. This is an explicit runtime
compatibility deviation from the official Torch2.3 pin. CPU rembg is sufficient:
the module must import, but explicit mask input skips rembg inference entirely.
The old ColorPeel environment is not modified.

```bash
ASSET=/var/tmp/colorpeel-r12user5/assets/marble_c_v1
/home/r12user5/miniforge3/envs/flux-smoke/bin/python -m venv \
  --system-site-packages "$ASSET/env"
MPY="$ASSET/env/bin/python"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export HF_HOME="$ASSET/hf_home" HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR=/var/tmp/colorpeel-r12user5/assets/pip-cache
"$MPY" -m pip install -r experiments/subject_marble_c_v1/requirements.txt
"$MPY" -B -m experiments.subject_marble_c_v1.assets --output "$ASSET"
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
# Select a NEW external output for every attempt, including failed attempts.
"$MPY" -B -m experiments.subject_marble_c_v1.run \
  --assets "$ASSET" --output "$FRESH_RUN" --gpu 2
```

Asset fetching pins all HuggingFace revisions. It downloads only necessary
SDXL base safetensors/components, fp16 Depth ControlNet, DPT, CLIP image encoder
and SDXL IPAdapter; both control MLPs come from the official Git checkout.
The base uses the official nonvariant weights, cast tofp16 at loading.
The official IPAdapter all-model download helper finds its required folders
prepopulated and does not download unused variants. All tracked official
files and downloaded model contents are hash-locked. Local-path substitution
and model CPU offload change storage/memory handling, not official editing math.
No custom attention processor is replaced for memory optimization.

## Outputs and checks

- `raw/*.png`: six untouched1024px MARBLE RGB outputs.
- `images/*.png`: distinct512px review candidates, raw resized with LANCZOS,
  with exact original RGB copied outside the shared binary mask. This is an
  explicit background postprocess, not proof of MARBLE background invariance.
- `source/`: exact original photograph/mask copies and shared DPT depth.
- `plan.json`, `runtime.json`, `generation_manifest.jsonl`, `status.json`:
  code/model/source revisions, calls, relative controls, proposed captions,
  raw/final hashes, metrics and failure state.
- `contact_sheet.png`, `index.html`: original+input mask and actual six RGBs
  labelled by relative M/R and quality flags. They are not training-approved.

Raw and composited background changes are measured separately. Structural
checks compare original against **raw512 before background composition**, using
tolerant gradient edges internally and near the input-mask boundary. They
are screening proxies affected by reflection changes; no output segmentation
or true silhouette IoU is claimed. HSV hue drift is measured only where both
source/output are chromatic; gloss/metallic changes can confound it. Luminance
quantiles and gradient statistics describe highlights but do not measure BRDF.
The full grid requires visual review of shape/parts, appearance, hue and borders.
Every row remains `training_eligible=false`; no concepts file or training set
is created. Failed candidates are retained and marked rather than used.
