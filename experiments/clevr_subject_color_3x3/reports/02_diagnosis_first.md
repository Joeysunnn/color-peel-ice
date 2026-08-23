# 02 — Diagnosis-first follow-up

- Status: **single-view diagnosis complete; human gate passed; multiview held-out next**
- Study: `clevr_subject_color_3x3`
- Method: `colorpeel_ice`
- Parent report: [`01_colorpeel_clevr_baseline.md`](01_colorpeel_clevr_baseline.md)
- Frozen training run: `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130816__clevr_subject_color_3x3__baseline__c8c874d__42`
- Frozen training commit: `c8c874d00318ae7c1df2265c8627787d316a1ce3`
- Completed evaluation adapter commit: `b059bd5e92cf1994581d8600111d3ed5830dc7d5`
- Conditional training config: [`../configs/train_cyan_initializer.yaml`](../configs/train_cyan_initializer.yaml)

## Decision and evidence hierarchy

The completed baseline is an immutable comparison anchor. Its checkpoint,
generated images, manifests, and scores must not be overwritten, resumed, or
silently regenerated. Follow-up work writes to new run directories and retains
the original prompt, seed, source-image, checkpoint, and commit provenance.

For semantic judgments about generated images, human visual review is the
primary evidence. Qwen3-VL, Grounded-SAM, and numerical color metrics are
secondary diagnostic evidence and may not overrule a documented human review
without adjudication. The present human review is qualitative and has no
per-seed ledger, so it is labeled **observed**, not **confirmed** and not used
to calculate an accuracy or error rate.

## Current human review

- **observed**: nine-grid prompts are mostly correct; approximately one or two
  outputs appeared exactly black.
- **observed**: subject-only cube and cylinder outputs were gray; sphere outputs
  were mostly gray with a small cyan minority.
- **observed**: color-only red outputs were mostly red and showed no consistent
  cube/sphere/cylinder leakage; gray outputs matched gray.
- **observed**: color-only cyan was less stable and some outputs had cube-like
  characteristics. This is the strongest current candidate for color-to-subject
  leakage, but the color-only prompt is outside the paired training template.
- **observed**: red and gray transfer were visually strong; cyan transfer was
  poor, with only occasional satisfactory examples.
- **observed, 2026-08-22 follow-up**: all learned subject tokens produced the
  requested subject when paired with literal red, cyan, or gray. The gray
  subject-only tendency is therefore attributed to the base-model default,
  not failure of the learned subject token to combine with color.
- **observed, 2026-08-22 follow-up**: in the trained-K/V cyan diagnostic, only
  the learned `<c2*>` condition was consistently poor. Literal `cyan`, `aqua`,
  `teal`, and `turquoise` were good under both trained K/V and vanilla SD 1.4;
  trained K/V was qualitatively a little more stable against variegated color.
  `turquoise` was the strongest trained-K/V candidate and was explicitly
  selected as the new `<c2*>` initializer.

The follow-up review inspected condition folders rather than completing the
randomized 540-row blind-review ledger. It is therefore qualitative and
unblinded. It closes the user-selection gate but is not reported as a blind
win rate or numerical accuracy.

- **observed, 2026-08-23 matched inference**: after changing only `<c2*>`
  initialization to `turquoise`, the user inspected the new inference folders
  and reported broad improvement across the protocol, with outputs now mostly
  meeting the requested behavior. This is the explicit human gate to proceed
  to multiview held-out validation; it is not converted into a numerical rate.

These observations revise the interpretation of the automated contingency
tables without deleting them. In particular, Qwen's sphere-only `other` labels
conflict with the human observation of mostly gray spheres. Single-axis prompts
remove a token role that was always present during training, and the color-only
form `a photo in <c*> color` does not name an object. Therefore single-axis bias
alone does not prove subject-color token entanglement.

## Stage gates

| Gate | Variant / artifact | Required evidence | Current status |
|---|---|---|---|
| 0. Freeze baseline | report 01 plus immutable run paths | training/evaluation commits, manifests, checkpoint path, generation path; no overwrite | **confirmed by existing records**; no new hashes claimed |
| 1. Human adjudication | per-item review ledger outside Git | image ID, prompt, seed, visible shape, visible color, black/ambiguous flags, reviewer note | **pending**; only qualitative summary exists |
| 2. Black-image diagnosis | `diagnostics_v1` | same checkpoint/prompt/seed; default SafetyChecker result and NSFW flag; separate explicitly acknowledged checker-disabled output; conditional FP32 finite checkpoint/pixel audit; distinct output directories | **completed**; checker-on 19/19 flagged and black, checker-off 19/19 finite and nonblack; FP32 not required |
| 3. Cyan diagnosis and initializer selection | 540-image cyan packet and 540 randomized single-image blind-review rows; then one selected initializer | trained/vanilla, candidate, template-family evidence; verified single-token candidates; explicit human approval of one candidate | **completed with protocol deviation**; 540/540 generated, folder-level review selected `turquoise`; randomized blind ledger not completed |
| 3b. Subject diagnosis | 75-image trained-K/V protocol | 3 shapes × seeds 42–46 × learned-only, natural-only, and learned+literal red/cyan/gray; per-image status and human review | **completed qualitatively**; 75/75 generated and all paired-color conditions accepted by the user |
| 4. Cyan initializer train | `cyan_initializer_ablation` | clean Git commit, dual smoke provenance, fresh training run, unchanged non-initializer settings | **completed** at commit `0959d1e`; both smokes passed and 1500/1500 loss rows were finite |
| 5. Matched evaluation | baseline protocol on the new checkpoint | fresh 900-row manifest, human review, secondary automated metrics, direct paired comparison | **human gate passed**; 900/900 valid, 0 black, 0 invalid; qualitative review reports broad improvement; secondary automated rescoring remains optional/pending |
| 6. Held-out multiview | `multiview_heldout_v1`; conditional `multiview_fold_{a|b|c}_seed{42|43|44}` | renderer provenance and held-out-view manifest before any training; separate render and train records | **next stage**; protocol ready, real renderer/output still under verification |
| 7. Factor-aware loss | no approved config | explicit loss definition, ablation control, code review, new method-risk approval | **conditional; not implemented or run** |
| 8. Natural multi-object test | no approved config | frozen prompt/image protocol and human rubric after synthetic diagnosis | **conditional; not implemented or run** |

Each gate closes only from named artifacts. A script, config, dry-run, or
process start is not evidence that a diagnostic or model run succeeded.

## Executed black-image diagnosis

Commit `ca3d313` was deployed through Git and passed 73 server tests. Under
FP16 with the default checker, all 19 exact-black source IDs were again flagged
and returned black. Under the same prompts, seeds, weights, steps, CFG, GPU,
and FP16 dtype, disabling only the checker recovered all 19 as finite, nonblack
images. The conditional FP32 stage was not needed. This closes the black-image
branch as SafetyChecker filtering rather than training instability.

The checker-disabled diagnostic completed with 540/540 cyan status rows and
75/75 subject status rows marked `ok`; the randomized review and condition-key
CSVs each contain 540 data rows. The user reviewed the organized condition
folders, accepted every learned-subject plus literal-color group, and selected
`turquoise` from the trained-K/V cyan candidates. Follow-up training and its
matched 900-image inference have now completed in fresh run directories.

## Approved single-variable training change

Gate 3 selected `turquoise`, which is locked directly in
`../configs/train_cyan_initializer.yaml`. The only intended training-semantic
change relative to the baseline is the `<c2*>` initializer. The following stay
fixed: nine CLEVR images, six modifier tokens, prompt template, SD 1.4,
Custom Diffusion K/V tuning, CAA weight 0.2, latent training mask, literal
official AdamW behavior, seed 42, 1500 steps, learning rate, scheduler, and all
other initializers.

The server-cached SD 1.4 tokenizer was checked with `AutoTokenizer` and
`local_files_only=True`. It encoded `cyan` as two IDs, `[1470, 550]`, while
`aqua`, `teal`, and `turquoise` each encoded as one ID: `[18613]`, `[22821]`,
and `[19899]`. The frozen baseline remains unchanged and retains its historical
initializer behavior. The follow-up validation rejects multi-piece
initializers instead of silently taking one piece.

Before candidate selection, `generate_cyan_diagnostic.py` defines 540 images:
ten nouns × seeds 42–44 × two prompt families, with 300 trained-K/V rows and
240 vanilla-SD rows. The trained rows cover learned `<c2*>` plus literal
`cyan`, `aqua`, `teal`, and `turquoise`; vanilla rows cover the four literal
words. `build_human_review.py` randomizes all 540 images into single-image
blind-review rows and writes a separate condition key. The reviewer records
color fidelity, prompt alignment, visual quality, invalid/artifact status, and
notes. These controls diagnose where current cyan failure enters the pipeline
and provide candidate evidence without retraining.

`generate_subject_diagnostic.py` separately defines 75 trained-K/V images:
three shapes × seeds 42–46 × five conditions. The conditions are learned
subject-only, natural-word subject-only, and the learned subject paired with
literal red, cyan, or gray. This tests whether the gray subject-only tendency
persists when color is explicitly supplied without changing the checkpoint.
Its server generation is queued behind the running cyan diagnostic; no result
is claimed yet.

Both cyan and subject diagnostic generators keep the default Stable Diffusion
SafetyChecker enabled. Disabling it is permitted only when both
`--disable-safety-checker` and `--acknowledge-safety-risk` are supplied; such
outputs remain isolated diagnostic evidence and never replace baseline images.

The config and protocol both lock the selection to `turquoise`, so runtime
environment variables cannot silently substitute another candidate. Training
other candidates as a sweep is not approved.

## Turquoise training result

The selected single-variable run completed at:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-233433__clevr_subject_color_3x3__cyan_initializer_ablation__0959d1e__42`.
Its launcher manifest reports `succeeded`, return code 0, and initializer list
`cube+sphere+cylinder+red+turquoise+gray`. The dedicated 2-step and 9-step
smokes both passed before full training. Full training wrote 1500 finite metric
rows and `checkpoint-1000`; every modifier-token exposure had a nonzero
gradient. `<c2*>` had 500 exposures, 500 nonzero-gradient steps, and an
initial-to-final L2 delta of `0.0231073`. Ordinary-vocabulary AdamW drift was
recorded for 49,408 rows and remained non-enforced, as required.

The run saved and reloaded final Custom Diffusion weights plus all six token
files before exiting successfully. Matched inference completed at
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-235330__clevr_subject_color_3x3__generate__388dc56__42`:
900/900 RGB 512×512 images were valid, with zero exact-black outputs. The user
then reported broad qualitative improvement and accepted progression to the
multiview held-out study.

## SafetyChecker diagnosis boundary

Disabling Stable Diffusion's SafetyChecker changes generation semantics and
weakens a safety control. It is allowed only as a targeted diagnostic after an
explicit `--acknowledge-safety-risk`, with the baseline source images kept
read-only and the rerun written to a new directory. Checker-disabled outputs
must not replace baseline images or enter headline baseline metrics.

The `scripts/methods/colorpeel_ice/rerun_black_images.py` entry point enforces
three ordered stages. `safety_flag` keeps FP16 and the default checker;
`disable_safety` keeps FP16 but requires both the disable and acknowledgement
flags; `fp32_finite` keeps the checker disabled, audits all learned tensors,
and checks finite FP32 pixels. Later stages accept only IDs still black in the
prior status file. The first two stages were executed in the run root recorded
above: 19/19 were flagged and black with the checker, then 19/19 were finite
and nonblack with only the checker disabled. No IDs remained for
`fp32_finite`, so it was not executed.

## Interpretation boundary

The current evidence supports paired-template success, a base-model gray
default for subject-only prompts, and an initializer-localized cyan failure.
Because learned `<c2*>` failed while literal colors worked with both vanilla
and trained K/V, and because the historical `cyan` initializer was silently
truncated from two tokenizer pieces, the initializer is the leading causal
explanation. The prospective `turquoise` single-variable retrain and matched
human review support that explanation. This closes the current single-view
gate, not the general disentanglement claim: multiview held-out validation is
next, while a new loss and natural multi-object evaluation remain conditional.

## Multiview preflight boundary

The three held-out folds, 16/4 view split, and three training seeds are
structurally valid. Before rendering, the multiview preparer was hardened to
use a dedicated turquoise base config, require an empty isolated output,
reject duplicate rendered images and missing camera/light/background
variation, and record held-out train-view and audit-view subsets separately.
These changes affect protocol integrity only; ColorPeel loss, CAA, K/V scope,
AdamW behavior, and training step order are unchanged.

No current tracked renderer satisfies the 180-view realization contract. The
actual 48-image data and closest modern renderer support object scale 1.3,
512 Cycles samples, and fixed object rotation. Official CLEVR supplies camera
jitter 0.5 and key/fill/back light jitter 1.0. It does not specify how to vary
the neutral world/ground background, so no range is selected without an
explicit experiment decision.
