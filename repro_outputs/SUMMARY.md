# ColorPeel-on-CLEVR reproduction summary

- GitHub fork: `https://github.com/Joeysunnn/color-peel-ice.git`
- Official upstream baseline: `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- Working branch: `repro/2026-08-22-colorpeel-diagnostics`
- Server-verified matched-inference commit: `388dc56c03fe69f846656a5d8e0e8897891cd896`
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
at `388dc56` after matched inference. The isolated `colorpeel017` environment was
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

## Diagnosis-first follow-up — single-view human gate passed

The baseline checkpoint is now the frozen comparison anchor. Its source run is
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130816__clevr_subject_color_3x3__baseline__c8c874d__42`
at training commit `c8c874d00318ae7c1df2265c8627787d316a1ce3`; the completed
evaluation adapters are anchored at `b059bd5e92cf1994581d8600111d3ed5830dc7d5`.
The ordered diagnostics, selected follow-up training, and matched inference are
complete. The matched semantic result is a qualitative human decision, not a
computed accuracy.

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
`[19899]`. The diagnostic completed 540/540 cyan images and 75/75 subject
images. Folder-level qualitative review found that learned subject tokens work
with literal colors, only learned `<c2*>` was poor among trained-K/V cyan
conditions, and the literal candidates worked similarly under trained K/V and
vanilla SD. The user selected `turquoise` as the strongest trained-K/V
initializer. The review was not completed through the randomized blind ledger,
so no blind win rate is claimed. The only intended training change is the
`<c2*>` initializer.

Both dedicated turquoise smokes passed. The full commit-`0959d1e` run completed
1500/1500 finite loss rows, saved `checkpoint-1000`, and produced nonzero
gradient/update evidence for all six modifier tokens. `<c2*>` was exposed and
updated on all 500 of its steps. The final Custom Diffusion and six token files
were saved and reloaded successfully. Matched inference then completed 900/900
valid RGB 512×512 images with zero exact-black outputs while explicitly
recording SafetyChecker disablement and acknowledgement. The user reported
broad, protocol-wide improvement and approved progression to multiview
held-out validation.

The ordered `diagnostics_v1` safety investigation is complete. Held-out
multiview rendering/training is next; any factor-aware loss and natural
multi-object evaluation remain conditional. SafetyChecker-disabled outputs are diagnostic-only,
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
but remained weak. The 540 cyan and 75 subject controls completed. The selected
`turquoise` initializer passed both smokes, full retraining, matched 900-image
inference, and the user's qualitative single-view gate.

The approved fixed-neutral renderer completed a real one-view V100 smoke and
the full 180-view run. Renderer commit `53a3a0e` produced 180/180 successful
records, zero failures, no residual partial directory, binary complementary
masks, and 20 unique RGB hashes per cell. Realization under validator commit
`de279ca` passed all profile, asset, scene, transform, hash, mask, split, and
staging checks. Each fold contains exactly 96 image-only training views and
three configs locked to `cube+sphere+cylinder+red+turquoise+gray`.

The 45-image contact sheet was visually inspected: all sampled objects were
complete and correctly shaped/colored, backgrounds remained neutral, and the
camera/light span showed no black or invalid images. The 180-row human-review
CSV remains unfilled, and all nine derived configs are explicitly
`pending_human_review`. No fold smoke, 1500-step training, held-out result, or
disentanglement-success claim exists. The final local and server suites each
passed 90 tests; every server code update was a clean GitHub `--ff-only`
fast-forward.
