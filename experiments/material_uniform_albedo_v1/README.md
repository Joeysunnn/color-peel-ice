# Uniform-albedo Material training

This independent dataset revision changes only Material supervision. It reads
the frozen Texture/Material v1 protocol, retains both target R/M pairs, and
preserves the old Texture data, training, checkpoints and inference results.
No existing experiment files are edited.

Each material has8 renders: sphere/cube x red/blue x soft-front/side lighting.
The shader has just Principled BSDF and Output nodes. Its spatially constant
Base Color is0.5 times the original controlled linear RGB; R/M stay fixed.
There are no image/procedural textures, normal maps, bump or displacement.
Lighting, camera, scene, geometries, scale,512px and64 Cycles samples reuse
the existing renderer settings. High/low illumination and reflections remain
valid material appearance; a uniform Albedo need not yield uniform image pixels.

The caption is `a photo of a {color} {shape} made of <M*> under {lighting}`.
The data validator checks all16 renders, fixed parameters, paired target masks,
source hashes, binary staged masks, captions and exact sampling membership.
There is no Texture dataset in this new run.

Four runs cover two materials x original fullrank/rank64 alpha64 LoRA.
They train from the same SD1.5 initialization, initializer`metal`, seed42,
embedding LR1e-5, fullrank K/V LR1e-5, rank64 K/V LR5e-5, batch1, constant LR,
no weight decay or extra attention/disentanglement loss. Other arguments match
the previous1000-step experiment. Each new trajectory saves500/750/1000,
and an automatic final audit exports the adapter with its matching token row
into `step500/`, `step750/`, `step1000/`.

Removing the two-texture dimension reduces each dataset from16 to8 images.
No images are duplicated to pad the count. With1000 fixed optimizer updates,
average exposure per image doubles. This is recorded as a comparability limit,
not described as a fully isolated texture-only pixel intervention. Uniform0.5
approximates the old patterns' average luminance; it is not exact image matching.

Develop/test/commit/push locally; server fetches and pulls`--ff-only`, checks a
clean revision and runs the existing environment. Use a fresh external output:

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$PY -B -m experiments.material_uniform_albedo_v1.run \
  --output "$COLORPEEL_OUTPUT_ROOT/material_uniform_albedo_v1/NEW_RUN" \
  --render-gpu 3 --train-gpus 1 2
```

The command renders and validates data, then trains and audits all four runs.
Both methods start in parallel; the second material queues behind the first.
`training/gpu_*.json` records job state, and each training directory contains
config, manifest, metrics and accelerator states under`checkpoints/checkpoint-N`.
`train_outputs/` records commands, configuration comparison and scientific limits.
Snapshots and paired inference exports are produced automatically after training;
this command does not generate diffusion inference images. Startup verification
is separate from completed training and semantic disentanglement assessment.
