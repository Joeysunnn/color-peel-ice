# MARBLE official sanity reproduction and mailbox diagnostics

Independent output root: `/var/tmp/colorpeel-r12user5/runs/subject_marble_diagnostics_v1`.
All existing experiment scripts, checkpoints and results remain immutable.

Actual completed runs and bounded findings: [RESULTS.md](RESULTS.md).
The subsequently authorized two mailbox joint grids are specified in
[JOINT_GRIDS.md](JOINT_GRIDS.md), with completed results in
[JOINT_RESULTS.md](JOINT_RESULTS.md).
The subsequent mailbox identity reconstruction test is specified in
[IDENTITY_TEST.md](IDENTITY_TEST.md), with the limited three-seed visual review in
[IDENTITY_RESULTS.md](IDENTITY_RESULTS.md). RGB changes are not treated as proof
of Albedo leakage; no dataset expansion or training was performed.
The authorized strength-only follow-up is specified in [STRENGTH_TEST.md](STRENGTH_TEST.md),
with completed results in [STRENGTH_RESULTS.md](STRENGTH_RESULTS.md).
The subsequent RGB-vs-gray initialization ablation is specified in [RGB_INIT_TEST.md](RGB_INIT_TEST.md),
with completed results in [RGB_INIT_RESULTS.md](RGB_INIT_RESULTS.md).

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

## Stage 2: mailbox size fix and attribute/reference controls

Stage1 nine images were completed and inspected before implementing Stage2.
The official example shows M/R responses but also creates gold decorations
at M=-20, so it is not a clean identity-preserving reference at all controls.

Add `--stage mailbox` to the command, using a different fresh output. This
uses the frozen original512 mailbox, existing mask, and exact old DPT cache
from the six-image pilot. The target is now resized to1024 **before** the
official ImageChops initialization. Mask interpolation remains official RGB
BICUBIC. The full material reference is explicitly the original512 photograph,
so resizing the target does not silently resize the CLIP reference too.

Two reference constructions retain the same512 canvas and foreground pixels:
full photo, versus background replaced by RGB127 gray using the frozen mask.
There is no crop, object enlargement or color normalization. This is a
background-reference ablation; object lettering/plaques remain in both.

Each reference uses eight controls: empty attribute dictionary, joint M0/R0,
M-only[-20,0,+20], and R-only[-1,0,+1]. Empty MLP does not mean identity;
all cases still use reference-guided diffusion. Zero controls are separate
because MLP offsets are not structurally constrained to zero. Two intentional
old512 controls (full-reference empty MLP and M-only-20) enable paired size
comparisons: **18 diagnostic RGBs total**, not a counterfactual training set.
All seed/sampling parameters remain unchanged.

Keep raw1024 outputs as primary evidence; do not restore backgrounds in this
study. Full-frame sheets, fixed subject crops and paired size comparisons
are saved separately. Input/model hashes and old Depth hash are verified
before/after; the old result directory is never written.

CPU regression, using the real pinned official initialization function:

```bash
export MARBLE_OFFICIAL_ROOT=/path/to/pinned/marble
python -B -m unittest discover -s tests/methods/colorpeel_ice -p test_marble_diagnostics.py -v
```

For a compact reference comparison plus descriptive pair metrics, verify all
18 raw hashes and write a NEW review directory, leaving generation unchanged:

```bash
python -B -m experiments.subject_marble_diagnostics_v1.review \
  --input "$COMPLETED_MAILBOX_RUN" --output "$FRESH_REVIEW"
```

No30-image expansion and no Subject training are part of this study.
