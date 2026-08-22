# ColorPeel-ICE research extension structure

This repository separates reusable method code from concrete experiment records. The method slug is `colorpeel_ice`; the first study slug is `clevr_subject_color_3x3`.

This document defines organization only. It does not import ICE model code, losses, stage names, data protocols, or algorithmic implementation.

## Tracked repository structure

```text
doc/
  PROJECT_STRUCTURE.md
  project-layout.md
experiments/
  README.md
  clevr_subject_color_3x3/
    README.md
    configs/
      baseline.yaml
      smoke_2step.yaml
      smoke_9step.yaml
      generate.yaml
      segment.yaml
      predict_qwen.yaml
      score_color.yaml
      score_clevr.yaml
    manifests/
      README.md
      clevr_3x3_manifest.json       maintained by the data-adapter workflow
    reports/
      README.md
      01_colorpeel_clevr_baseline.md
literature/
  README.md
  _template.md
  notes/
    2024_butt_colorpeel.md
    2025_cendra_ice.md
scripts/
  launch/colorpeel_run.py
  methods/colorpeel_ice/
  setup/setup_colorpeel017.sh
src/
  methods/colorpeel_ice/
  train/                         official training code plus audited token fix
tests/
  test_experiment_runner.py
  methods/colorpeel_ice/
```

Large or generated artifacts never belong in these directories. Checkpoints, logs, generated images, masks, predictions, scores, and figures live under `$COLORPEEL_RUN_ROOT/<study_slug>/<run_id>/` outside the Git working tree.

## Method versus study

- `colorpeel_ice` names the implementation family. Future method-specific code should use a matching method directory rather than adding unrelated top-level scripts.
- `clevr_subject_color_3x3` names one research question and owns only tracked configuration, immutable small manifests, and evidence-linked reports.
- A new research mechanism receives a new method slug. A new dataset/task or research question receives a new study slug. A seed, learning rate, step count, or sampler change is a variant inside an existing report.
- Code is shared only after two methods demonstrably use it. Shared boundaries must be documented rather than inferred from placement.

## Evidence boundary

Every experimental statement uses exactly one label:

- **confirmed**: directly supported by a named tracked config, manifest, run manifest, metric file, or scorer output;
- **observed**: human inspection or verbal observation without complete machine-readable evidence;
- **pending**: not run, incomplete, blocked, or missing the required artifact.

All current `clevr_subject_color_3x3` findings are `pending`. Paper results are external evidence and never confirm this repository's experiment.
