# Material eight-token training smokes

Release commit: `6fdd889edbba870842bf979d3f7e21c3cc1dbbf5`

Both independent GPU-3 smokes completed with return code 0 and passed
`src.train.training_audit`. Losses were finite, Custom Diffusion weights and
all eight token files were nonempty, and post-save reload succeeded.

## Two-step paired-material smoke

Run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260828-205100__clevr_subject_color_material_3x3x2__smoke2-material-pair__6fdd889__42`

- metric rows: 2
- total-loss range: `0.0345137641`--`0.6939486265`
- exposures: `<s1*>:2`, `<c1*>:2`, `<m1*>:1`, `<m2*>:1`; other tokens 0
- all four exposed tokens had nonzero gradients and positive embedding deltas
- ordinary-vocabulary drift: 49,408 rows, max L2 `2.8148179e-7`, record-only

## Eighteen-step full-grid coverage smoke

Run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260828-205200__clevr_subject_color_material_3x3x2__smoke18-full-grid-coverage__6fdd889__42`

- metric rows: 18
- total-loss range: `0.0231324770`--`1.6379591227`
- each subject/color token exposure: 6; each material token exposure: 9
- all eight tokens had nonzero gradients on every exposure and positive deltas
- ordinary-vocabulary drift: 49,408 rows, max L2 `2.5333359e-6`, record-only

These are startup/coverage checks, not model-quality or disentanglement
results. They release the locked full-grid seed-42 baseline but do not release
Fold training.
