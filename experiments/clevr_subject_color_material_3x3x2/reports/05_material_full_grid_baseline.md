# Material full-grid baseline: human gate

## Outcome

The seed-42 full 18-cell baseline completed 1500 optimization steps on GPU 3.
All 1500 reconstruction, CAA, and total-loss records were finite. The run saved
`checkpoint-1000`, final Custom Diffusion weights, all eight modifier-token
artifacts, and `embedding_update_audit.json`; the launcher exited successfully.

The matched full-grid inference completed 360/360 512x512 RGB images for 18
complete subject/color/material prompts and seeds 42-61. Every status row was
successful and every recorded image SHA-256 matched the realized file.

## Runs

- Training:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260828-210000__clevr_subject_color_material_3x3x2__full_grid_seed42__29d4131__42`
- Generation:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260828-211300__clevr_subject_color_material_3x3x2__material_full_grid_generate_seed42__29d4131__42`
- Human-review bundle:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260829-142200__clevr_subject_color_material_3x3x2__material_full_grid_human_bundle_seed42__d48ffa8__42`

## Training audit

- Modifier exposure/nonzero-gradient steps:
  - `<s1*>`: 540/540; delta 0.0155934431
  - `<s2*>`: 480/480; delta 0.0185688641
  - `<s3*>`: 480/480; delta 0.0245901607
  - `<c1*>`: 512/512; delta 0.0192836132
  - `<c2*>`: 508/508; delta 0.0240884256
  - `<c3*>`: 480/480; delta 0.0232685339
  - `<m1*>`: 752/752; delta 0.0209641326
  - `<m2*>`: 748/748; delta 0.0202173349
- Literal official AdamW drift was recorded for 49,408/49,408 ordinary
  vocabulary rows. It remains observation-only and is not a failure condition.

## Interrupted-generation recovery

The first foreground generation process was killed when the SSH connection was
reset after 76 images. Those 76 files were retained. Each was checked against
the immutable manifest for ID, 512x512 RGB validity, model/protocol
fingerprints, and a newly calculated SHA-256 before a recovery status ledger
was written. The same run and Git revision then resumed with `--resume` under
`nohup`; it skipped the 76 verified files and completed the remaining 284.
`resume_recovery_audit.json` records this operational recovery. No scientific
parameter, prompt, seed, checkpoint, or output file was replaced.

## Human-review artifacts

- `human_review.csv`: 360 randomized rows.
- `red_sphere_material_regression.csv`: 40 rows.
- `material_pair_sheets/`: nine shape/color sheets with same-seed metal/rubber
  pairs.
- `human_gate_decision.json`: `pending_human_review` and
  `heldout_training_authorized=false`.

The bundler direct-script import defect was fixed in commit `d48ffa8`; the
server then passed all 134 tests. The failed pre-fix bundle run is retained as
evidence and was not overwritten. No held-out Fold training has started.
