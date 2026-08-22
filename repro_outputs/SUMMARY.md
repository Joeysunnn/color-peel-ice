# ColorPeel-on-CLEVR reproduction summary

- GitHub fork: `https://github.com/Joeysunnn/color-peel-ice.git`
- Official upstream baseline: `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- Working branch: `repro/2026-08-22-colorpeel-diagnostics`
- Server-verified evaluation commit: `b059bd5e92cf1994581d8600111d3ed5830dc7d5`
- Baseline status: `completed_with_reported_evaluation_failures`
- Baseline code/test status: `success` (46 isolated final tests; compile and JSON checks
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

## Diagnosis-first follow-up — pending

The baseline checkpoint is now the frozen comparison anchor. Its source run is
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130816__clevr_subject_color_3x3__baseline__c8c874d__42`
at training commit `c8c874d00318ae7c1df2265c8627787d316a1ce3`; the completed
evaluation adapters are anchored at `b059bd5e92cf1994581d8600111d3ed5830dc7d5`.
No follow-up diagnostic, render, training, or evaluation result is claimed by
this documentation update.

Human review is primary for image semantics and currently remains qualitative:
the grid was mostly correct with roughly one or two black images; subject-only
outputs were predominantly gray; color-only red and gray were visually
faithful; cyan was less stable and sometimes cube-like; and cyan transfer was
substantially weaker than red/gray. Because there is no per-seed review ledger,
these are `observed` findings, not computed rates. Qwen and color metrics remain
secondary evidence; notably, Qwen's sphere-only `other` labels disagree with
the human description of mostly gray spheres.

The approved conditional training variant is
`cyan_initializer_ablation`, configured by
`experiments/clevr_subject_color_3x3/configs/train_cyan_initializer.yaml`.
The cached SD 1.4 tokenizer directly encoded `cyan` as two IDs `[1470, 550]`,
but `aqua`, `teal`, and `turquoise` as the single IDs `[18613]`, `[22821]`, and
`[19899]`. No replacement has been chosen. The pending cyan diagnostic contains
540 images (300 trained-K/V and 240 vanilla-SD) followed by 540 randomized
single-image blind-review rows. Human review must select one candidate before
training; the only intended training change is the `<c2*>` initializer.

The ordered `diagnostics_v1` safety investigation, held-out multiview rendering
and training, any factor-aware loss, and natural multi-object evaluation are
all pending or conditional. SafetyChecker-disabled outputs are diagnostic-only,
require explicit acknowledgement, and may not replace baseline outputs.

The diagnosis-first implementation passed 73 local tests plus compile, JSON,
YAML, and dry-run manifest checks. The local all-in-one pytest process requires
`KMP_DUPLICATE_LIB_OK=TRUE` because this Windows Anaconda installation loads
incompatible OpenMP runtimes through PyTorch and NumPy; isolated suites pass
without it. This is a local test-runner condition and is not exported to server
training or inference.

Commit `ca3d313c4d081bcdec5fda6979b05c4fde3415c0` was pushed and deployed to the
clean server checkout through Git. The server repeated all 73 tests in 5.40
seconds. In the ordered black-image diagnostic, all 19 source-black IDs were
again flagged by the default checker and returned black under FP16; disabling
only the checker recovered 19/19 finite, nonblack images under the same FP16
settings. FP32 fallback was therefore not required. These black outputs are a
SafetyChecker filtering artifact, not evidence of diffusion/VAE instability.

The source-aware transfer rerun retained 600 rows with 588 valid masks. Median
source references from the nine GT masks were red `[83,35,35]`, cyan
`[38,91,91]`, and gray `[58,58,58]`. Cyan mean 50%-pixel DeltaE improved from
66.0792 against nominal RGB to 42.5543 against the rendered-metal reference,
but remained weak. The 540 cyan and 75 subject controls are currently
generating on GPU 3; no initializer has been selected and no retraining has
begun.
