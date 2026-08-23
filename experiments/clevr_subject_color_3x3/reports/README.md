# Study report index

| ID | Hypothesis | Fixed config | Status | Run path | Evidence |
|---|---|---|---|---|---|
| [01](01_colorpeel_clevr_baseline.md) | ColorPeel can be evaluated on the locked CLEVR 3×3 six-token grid without ICE method additions | baseline plus two smoke and post-generation stage configs | completed | grid joint 93.89%; 588/600 transfer masks | mixed |
| [02](02_diagnosis_first.md) | Diagnose black outputs, evaluator disagreement, and cyan instability before changing training semantics | frozen report-01 baseline; conditional `train_cyan_initializer.yaml` | pending | none; follow-up runs not started | qualitative human review only |
| [03](03_multiview_orbit_v2.md) | Replace weak camera translation variation with object-centered orbit viewpoints without changing training semantics | locked v2 profile/protocol; v1 preserved | implementation complete; runtime gate pending | no Fold training | tests plus real Blender smoke required |

One report covers one hypothesis and its fair controls. Setting-only changes belong in the report's variant table; do not create a separate report solely for a seed, learning rate, step count, or sampler setting.
