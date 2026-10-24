# MARBLE official sanity reproduction and mailbox diagnostics

Independent output root: `/var/tmp/colorpeel-r12user5/runs/subject_marble_diagnostics_v1`.
All existing experiment scripts, checkpoints and results remain immutable.

## Stage 1: official notebook example

Use the pinned official checkout and model receipt already prepared by
`experiments.subject_marble_c_v1.assets`. Reproduce the full notebook nine-case
grid: native1024 RGB toy_car context, native RGBA metal_bowl reference,
M=[-20,0,+20], R=[-1,0,+1], seed42,30steps and unchanged official defaults.
The official rembg algorithm and DPT extraction are computed once and saved;
passing these cached conditions is an explicit resource adaptation. The
repository's pre-rendered toy-car depth is not substituted. Original1024
RGBs are retained without background copying. A contact sheet includes the
actual context, reference, mask and depth. No image is training-approved.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export HF_HOME=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/hf_home
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python -B \
  -m experiments.subject_marble_diagnostics_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$FRESH_RUN" --gpu 2
```

The first output path must be fresh and outside all code/assets. Official rembg
may download its default U2Net weights to the separate assets subdirectory
`diagnostics_rembg`; it does not modify official tracked files. Existing
Python3.11/Torch2.5.1 environment and CPU offload are disclosed deviations.
The official notebook has no saved generated PNGs; this is a visual sanity
reproduction, not a claim of pixel equality or full-paper reproduction.

Inspect Stage1 before implementing/running the mailbox size fix and controls.
No30-image expansion and no Subject training are part of this study.
