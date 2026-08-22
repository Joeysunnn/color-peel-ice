# CLEVR subject-color 3×3 study

- Study slug: `clevr_subject_color_3x3`
- Method slug: `colorpeel_ice`
- Baseline status: **completed with explicit evaluation failures**
- Follow-up status: **diagnosis-first plan pending; no new training claimed**
- Baseline: ColorPeel commit `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- Dataset root: `/home/r12user5/Documents/Jiawei/papers/ICE/datasets/clevr_basic_neutral_stage1_gt`
- Run root: `$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/<run_id>/`

## Research question

Can the official ColorPeel training mechanism learn three subject tokens and three color tokens on the locked CLEVR metal 3×3 grid while exposing subject/color leakage with predeclared evaluation outputs?

This is a ColorPeel-on-CLEVR reproduction. ICE supplies the problem context only. ICE stages, losses, masks, token parameterizations, and other method content are excluded.

## Locked concepts

| Token | Meaning | Initializer |
|---|---|---|
| `<s1*>` | cube | `cube` |
| `<s2*>` | sphere | `sphere` |
| `<s3*>` | cylinder | `cylinder` |
| `<c1*>` | red | `red` |
| `<c2*>` | cyan | `cyan` |
| `<c3*>` | gray | `gray` |

Material is fixed to `metal` and has no learned token. Every prompt uses `a photo of <subject-token> shape in <color-token> color`.

## Locked sample grid

| Shape | Red | Cyan | Gray |
|---|---|---|---|
| cube | `003_cube_red_metal` | `013_cube_cyan_metal` | `001_cube_gray_metal` |
| sphere | `019_sphere_red_metal` | `029_sphere_cyan_metal` | `017_sphere_gray_metal` |
| cylinder | `035_cylinder_red_metal` | `045_cylinder_cyan_metal` | `033_cylinder_gray_metal` |

The source inventory was verified as 48 samples (`3 shapes × 8 colors × 2
materials`). Images and GT masks are 512×512 and masks are binary. The
run-specific audit path is named in the study report and
`repro_outputs/DATASET_AUDIT.md`. GT masks remained audit/evaluation assets and
did not enter the training loss.

## Baseline protocol

The tracked baseline is [`configs/baseline.yaml`](configs/baseline.yaml). It locks:

- `CompVis/stable-diffusion-v1-4`, Diffusers `v0.17.0`, 512 px;
- Custom Diffusion cross-attention K/V optimization plus modifier embeddings;
- batch 1, 1500 steps, LR `1e-5`, scaled LR, constant scheduler, zero warmup;
- seed 42, CAA/cosine weight 0.2, gradient accumulation 1;
- AdamW β₁ 0.9, β₂ 0.999, weight decay 0.01, ε `1e-8`;
- max gradient norm 1.0, two data workers, checkpoint interval 1000;
- mixed precision `no`, xFormers off, prior preservation off, resume off.

Mixed precision `no` is an explicit run lock because the official repository delegates this choice to Accelerate. The other values follow the inspected official launcher/parser unless the config says otherwise.

## Locked AdamW behavior

Status: **confirmed** by the real 48-sample audit and nine-image staging.

The official code passes the full text-embedding parameter to AdamW with weight decay 0.01. Consequently, ordinary vocabulary rows may drift through literal decoupled AdamW decay even when their gradients are zero. This study deliberately preserves that official behavior:

- policy: `literal_official_adamw_decay_allowed`;
- do not restore or freeze ordinary vocabulary values after `optimizer.step()`;
- record ordinary-vocabulary drift as an observation;
- do not fail a smoke or full run solely because this drift is nonzero;
- keep the behavior visible as a comparability caveat.

This decision removes the former training blocker. It does not change the requirement that every exposed modifier token demonstrate an actual learning signal in the nine-step smoke.

## Two real training smokes

Launcher `--dry-run` is preflight only and does not count as either smoke.

1. [`smoke_2step.yaml`](configs/smoke_2step.yaml) uses only the first two locked samples: `003_cube_red_metal` and `013_cube_cyan_metal`. Expected exposures are `<s1*>:2`, `<c1*>:1`, `<c2*>:1`, and zero for `<s2*>`, `<s3*>`, `<c3*>`. Unseen tokens have no gradient or delta requirement.
2. [`smoke_9step.yaml`](configs/smoke_9step.yaml) uses every grid cell exactly once. Every modifier token must have exposure count 3, at least one nonzero gradient observation, and a nonzero final embedding delta.

Both are independent real training runs with distinct immutable run directories. Each must record finite loss observations, exit status, token observations, ordinary-vocabulary drift, saved artifacts, and reload outcome. Neither smoke is a paper-result reproduction or a substitute for the 1500-step run.

## Evaluation outputs

The planned protocol uses seeds 42–61 and records:

- nine-grid reconstruction, subject-only, color-only, and ten color-transfer prompt families;
- official ColorPeel color metrics separately from CLEVR-specific diagnostics;
- shape, color, and joint accuracy from frozen deterministic predictions;
- `subject token × predicted color` and `color token × predicted shape` contingency tables;
- explicit segmentation and prediction failure rows.

Grounded-SAM and Qwen3-VL are independent tracked post-generation stages, configured by `segment.yaml` and `predict_qwen.yaml`. Each owns its own immutable run directory, command, environment/model provenance, per-item output, and failure rows. `score_color.yaml` may consume Grounded-SAM masks only after segmentation completes; `score_clevr.yaml` may consume Qwen prediction JSONL only after prediction completes.

Training/generation use `colorpeel017`; Grounded-SAM and color scoring use the existing `/home/r12user5/miniforge3/envs/ice/bin/python`; Qwen3-VL and CLEVR scoring use `/home/r12user5/miniforge3/envs/ice-vlm/bin/python`. These two existing ICE environments are read-only dependencies for this workflow and must not be modified. Grounded-SAM/Qwen models use `local_files_only`; cache absence is an explicit failed stage with complete per-item failure JSONL and nonzero exit.

No custom numerical threshold defines “entanglement solved.” The completed run
shows 93.89% grid joint accuracy alongside strong single-axis contingency bias;
the report records both without converting them into a new success threshold.

## Diagnosis-first follow-up

The completed baseline checkpoint and all baseline artifacts are frozen as the
comparison anchor. Follow-up diagnostics and ablations use new immutable run
directories and may not overwrite, resume, or silently regenerate baseline
outputs. See [`reports/02_diagnosis_first.md`](reports/02_diagnosis_first.md).

Human visual review is primary for generated-image semantics; Qwen3-VL,
Grounded-SAM, and color metrics remain secondary diagnostics. The current
human review is qualitative and lacks a per-seed ledger, so it is `observed`
rather than `confirmed`. It identifies cyan instability and occasional
cube-like color-only cyan outputs as hypotheses to diagnose, not proof of
token entanglement.

The only approved conditional training change is the `<c2*>` initializer in
[`configs/train_cyan_initializer.yaml`](configs/train_cyan_initializer.yaml).
`COLORPEEL_CYAN_INITIALIZER` must be selected by a recorded diagnostic from
`aqua`, `teal`, or `turquoise`; it is intentionally not fixed in advance. CAA,
AdamW, the training mask, prompt, dataset, seed, steps, and all other
initializers remain unchanged.

`diagnostics_v1` is a targeted black-image/SafetyChecker diagnosis. A
checker-disabled rerun requires explicit safety acknowledgement, writes to a
separate directory, and never replaces baseline outputs. Multiview rendering
or training (`multiview_heldout_v1`), a factor-aware loss, and natural
multi-object evaluation remain pending or conditional and have no completed
artifacts.

## Records

- Manifest policy: [`manifests/README.md`](manifests/README.md)
- Report index: [`reports/README.md`](reports/README.md)
- Baseline report: [`reports/01_colorpeel_clevr_baseline.md`](reports/01_colorpeel_clevr_baseline.md)
- Diagnosis-first report: [`reports/02_diagnosis_first.md`](reports/02_diagnosis_first.md)
- Literature: [ColorPeel](../../literature/notes/2024_butt_colorpeel.md), [ICE](../../literature/notes/2025_cendra_ice.md)
