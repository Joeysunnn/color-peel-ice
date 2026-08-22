# Experiment registry

Create one lower-case `study_slug` directory per research question. Tracked study directories contain definitions and reviewed records only:

```text
experiments/<study_slug>/
  README.md
  configs/
  manifests/
  reports/
```

Runtime artifacts belong below `$COLORPEEL_RUN_ROOT/<study_slug>/<run_id>/`, never in this registry.

## Record levels

| Change | Record |
|---|---|
| Research mechanism | new method slug and method directory |
| Dataset, task, or research question | new study directory |
| One hypothesis and its fair controls | numbered report in `reports/` |
| Seed, LR, steps, sampler, or another setting | variant row in the existing report plus tracked YAML |

Before a run, record the hypothesis, fixed protocol, variants, evaluation outputs, known conflicts, and acceptance evidence. After a run, record the absolute immutable run path, checkpoint, generation/evaluation configs, and the source file for every number.

Use only `confirmed`, `observed`, or `pending` evidence labels. Preserve superseded observations rather than rewriting history.

## Registered studies

| Study | Method | Question | Status |
|---|---|---|---|
| [`clevr_subject_color_3x3`](clevr_subject_color_3x3/README.md) | `colorpeel_ice` | Can the official ColorPeel mechanism be reproduced on the locked CLEVR 3×3 subject/color grid without ICE method additions? | pending |
