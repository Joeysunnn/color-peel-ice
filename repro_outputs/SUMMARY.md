# ColorPeel-on-CLEVR reproduction summary

- GitHub fork: `https://github.com/Joeysunnn/color-peel-ice.git`
- Official upstream baseline: `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- Working branch: `repro/2026-08-21-colorpeel-clevr`
- Server-verified evaluation commit: `b059bd5e92cf1994581d8600111d3ed5830dc7d5`
- Overall status: `completed_with_reported_evaluation_failures`
- Code/test status: `success` (46 isolated final tests; compile and JSON checks
  passed; `git diff --check` clean apart from line-ending notices)
- Training status: `success` (2-step, 9-step, and 1500-step)
- Generation/evaluation status: `success` (900/900 generated and valid;
  Grounded-SAM 588/600 masks; Qwen 300/300 predictions)

The 3×3 CLEVR adaptation, six-token gradient-boundary correction, data adapter,
auditable run launcher, deterministic 900-item generation protocol, scorers,
project structure, experiment registry, and literature records are pushed to
the fork. The dual-smoke observation contract and independent Grounded-SAM and
Qwen stages were deployed from the same clean Git history.

The server checkout at `/home/r12user5/Documents/Jiawei/colorpeel/` was created
from the GitHub branch and updated only with `git pull --ff-only`; it was clean
at `c8c874d` when runs launched. The isolated `colorpeel017` environment was
recreated and frozen. Existing `ice` and `ice-vlm` environments and shared
model caches were inspected read-only and not modified.

The real 48-sample audit passed and the nine-image loader tree contains only
image links. Both smokes passed their exact coverage contracts. The full
1500-step run completed with 1500 finite loss rows, a step-1000 checkpoint,
nonzero updates for all six tokens, final Custom Diffusion/token weights, and a
successful six-token reload. Literal official AdamW drift affected all 49,408
ordinary rows and was recorded without restoration or failure threshold.

The 900-row generation manifest is locked to 180 grid, 60 subject-only, 60
color-only, and 600 transfer items with seeds 42–61. All 900 images passed
decode, native-RGB, and 512×512 checks. Grounded-SAM produced 588 valid masks
and explicitly reported 12 failures; color metrics contain the same 588/12
split. Qwen3-VL returned 300/300 valid fixed-JSON predictions. On the 180 grid
images, shape accuracy was 94.44%, color accuracy 93.89%, and joint accuracy
93.89%.

The single-axis contingency tables still show strong axis-dependent output
biases, so this evidence does not authorize the statement that entanglement is
“solved.” It supports a successful ColorPeel-on-CLEVR training/generation
reproduction plus an auditable, mixed disentanglement diagnosis.
