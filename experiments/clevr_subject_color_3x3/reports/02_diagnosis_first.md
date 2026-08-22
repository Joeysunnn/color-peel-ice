# 02 — Diagnosis-first follow-up

- Status: **pending; baseline frozen and no follow-up model run claimed**
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
| 2. Black-image diagnosis | `diagnostics_v1` | same checkpoint/prompt/seed; default SafetyChecker result and NSFW flag; separate explicitly acknowledged checker-disabled output; final FP32 finite checkpoint/pixel audit; distinct output directories | **pending** |
| 3. Cyan diagnosis and initializer selection | 540-image cyan packet and 540 randomized single-image blind-review rows; then one selected initializer | trained/vanilla, candidate, template-family evidence; completed human CSV; verified single-token candidates; explicit human approval of one candidate | **pending**; tokenizer evidence exists, but no packet or candidate decision |
| 3b. Subject diagnosis | 75-image trained-K/V protocol | 3 shapes × seeds 42–46 × learned-only, natural-only, and learned+literal red/cyan/gray; per-image status and human review | **pending; not run** |
| 4. Cyan initializer train | `cyan_initializer_ablation` | gates 1–3 closed, clean Git commit, dry-run provenance, fresh training run, unchanged non-initializer settings | **conditional / not run** |
| 5. Matched evaluation | baseline protocol on the new checkpoint | fresh 900-row manifest, human review, secondary automated metrics, direct paired comparison | **conditional / not run** |
| 6. Held-out multiview | `multiview_heldout_v1`; conditional `multiview_fold_{a|b|c}_seed{42|43|44}` | renderer provenance and held-out-view manifest before any training; separate render and train records | **pending; no render or training claimed** |
| 7. Factor-aware loss | no approved config | explicit loss definition, ablation control, code review, new method-risk approval | **conditional; not implemented or run** |
| 8. Natural multi-object test | no approved config | frozen prompt/image protocol and human rubric after synthetic diagnosis | **conditional; not implemented or run** |

Each gate closes only from named artifacts. A script, config, dry-run, or
process start is not evidence that a diagnostic or model run succeeded.

## Approved single-variable training change

If gate 3 selects a candidate, set `COLORPEEL_CYAN_INITIALIZER` to exactly one
of `aqua`, `teal`, or `turquoise` and use
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
It is pending and has no server output.

Both cyan and subject diagnostic generators keep the default Stable Diffusion
SafetyChecker enabled. Disabling it is permitted only when both
`--disable-safety-checker` and `--acknowledge-safety-risk` are supplied; such
outputs remain isolated diagnostic evidence and never replace baseline images.

The config intentionally contains `${COLORPEEL_CYAN_INITIALIZER}` rather than
a chosen word. It must remain non-runnable until the diagnostic packet is
reviewed and an explicit record selects one allowed candidate. Training all
three candidates and choosing the best result afterward would be an undeclared
sweep and is not approved.

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
prior status file. The entry point exists, but none of these stages is recorded
as executed in this handoff.

## Interpretation boundary

The current evidence supports a diagnosis of paired-template success plus
single-axis context dependence. It does not yet establish that subject and
color tokens are fully disentangled, nor that cyan-to-cube behavior is caused
by the initializer. The initializer ablation may test that hypothesis only
after the diagnostic gates close. Multiview data, a new loss, and natural
multi-object evaluation are later conditional studies and cannot be used to
explain the frozen baseline retroactively.
