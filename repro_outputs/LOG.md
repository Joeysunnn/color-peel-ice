# Reproduction log

## 2026-08-21 — repository and protocol audit

- Read the full official ColorPeel code at baseline commit `021f5c7` and the
  ColorPeel/ICE papers.
- Locked the CLEVR metal grid, six tokens, prompts, official parameters, and
  evaluation seeds in the approved plan.
- Direct server inspection established the reported 48-sample inventory.
- Found the official final-modifier gradient off-by-one defect and separately
  identified literal AdamW weight-decay drift on ordinary vocabulary rows.

## 2026-08-22 — server rollback

- Removed only the task-created conda environment
  `/home/r12user5/miniforge3/envs/colorpeel017` (about 4.0 GB).
- Removed only the task-created temporary directory
  `/tmp/colorpeel_data_adapter_test` after listing its three files.
- Did not clear shared pip/conda caches and did not touch ICE.
- Confirmed the superseded target
  `/home/r12user5/Documents/Jiawei/papers/color-peel-clevr` never existed.
- Confirmed `/home/r12user5/Documents/Jiawei/colorpeel/` existed and was empty
  before the GitHub clone.

## 2026-08-22 — local GitHub-first implementation

- Set `origin=https://github.com/Joeysunnn/color-peel-ice.git` and retained
  `upstream=https://github.com/moatifbutt/color-peel.git`.
- Verified both `main` branches pointed to official commit `021f5c7` before
  applying changes.
- Added method/study separation, tracked configs/manifests/reports, literature
  notes, external run-root contract, and a provenance-recording launcher.
- Local verification succeeded for the earlier implementation snapshot, before
  the current smoke-observation and independent-stage changes.
- Created implementation commit `41d752a9d8e8b3a5ab711db90990ab28e4f58000`
  and pushed the reproduction branch to the fork.

## 2026-08-22 — server checkout from GitHub

- Cloned the fork branch directly into
  `/home/r12user5/Documents/Jiawei/colorpeel/`.
- The checkout was later fast-forwarded through GitHub and is currently
  verified clean on branch `repro/2026-08-21-colorpeel-clevr` at HEAD
  `e6c57d1ba9074db50f07a32cb56bebaffcc44876`.
- The newer pre-run handoff changes covered by the 44-test suite have not yet
  been fast-forwarded to the server.
- No environment, model, data staging, training, or evaluation command was run.

## 2026-08-22 — optimizer decision and next evidence boundary

Literal official AdamW updates zero-gradient embedding rows through decoupled
weight decay. The locked policy is now
`literal_official_adamw_decay_allowed`: keep this behavior, record ordinary
vocabulary drift, do not restore those values, and do not fail a run solely for
nonzero drift.

No real smoke has run. Preflight dry-runs do not count as smokes. The next two
training records are independent: a two-step first-two-sample smoke where
unseen tokens have no gradient requirement, and a nine-step full-grid smoke
where every modifier token must have exposure 3, at least one nonzero-gradient
step, and nonzero final delta.

Grounded-SAM and Qwen3-VL remain independent post-generation stages. Neither
may be inferred from generation or from downstream scorer availability.
The current changes have not yet received a complete local test rerun; no old
test count is reused as evidence for them.

Subsequent pre-run handoff verification completed with `44 passed` in the
isolated pytest suite. This is local code/config evidence only; no current
server run was performed and the server has not yet fast-forwarded to this
handoff.

## 2026-08-22 — GitHub-first deployment and environment verification

- Pushed the complete pre-run workflow as commit
  `6adca6dab9ad177c58a99b6ab26662cc92e8c140`, then fast-forwarded the clean
  server checkout exclusively with `git pull --ff-only`.
- Built `/home/r12user5/miniforge3/envs/colorpeel017` with the locked runtime;
  `pip check` reported no broken requirements and the environment freeze is at
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/environment/colorpeel017-pip-freeze.txt`.
- Verified the official Diffusers `v0.17.0` tag and loaded
  `CompVis/stable-diffusion-v1-4` successfully with
  `local_files_only=True` after its authorized download.
- Ran the complete server test suite: `44 passed` before the compatibility
  patch. The patch-specific training audit suite subsequently passed 14 tests.
- The real 48-sample audit and nine-image staging passed; see
  `DATASET_AUDIT.md`.

## 2026-08-22 — zero-step compatibility failure and resolution

- The first two-step attempt at commit `6adca6d` stopped before model loading
  and before any optimization step: pinned Accelerate 0.20.3 rejected the
  newer `logging_dir` argument to `Accelerator`.
- The minimal logging-only correction moved that value into the already-used
  `ProjectConfiguration`. No loss, optimizer, gradient, data-order, or update
  semantics changed.
- Pushed commit `c8c874d00318ae7c1df2265c8627787d316a1ce3` and again deployed only
  by `git pull --ff-only`. Diagnosis and rerun evidence are in
  `debug_outputs/`.

## 2026-08-22 — real smoke results

- Two-step run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130559__clevr_subject_color_3x3__smoke2-first-two__c8c874d__42`.
  Validator status `passed`; two finite metric rows; all seven final weight
  artifacts nonempty and all six tokens reloaded. Ordinary-vocabulary drift
  was descriptive only: 49,408 changed rows, mean L2
  `9.022458868912508e-08`, max `2.81481788988458e-07`.
- Nine-step run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130706__clevr_subject_color_3x3__smoke9-full-grid__c8c874d__42`.
  Validator status `passed`; nine finite metric rows; every token had exposure
  3, three nonzero-gradient steps, and nonzero final delta. All seven weight
  artifacts were nonempty and all six tokens reloaded. Ordinary-vocabulary
  drift: 49,408 changed rows, mean L2 `4.060104572545242e-07`, max
  `1.2666679367612232e-06`, not enforced.

## 2026-08-22 — full training result

- Run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130816__clevr_subject_color_3x3__baseline__c8c874d__42`.
- Manifest status `succeeded`, return code 0, exactly 1500 finite metric rows,
  total-loss range `0.004082555416971445` to `3.183166265487671`.
- Step-1000 accelerator checkpoint exists. Final Custom Diffusion weights and
  all six token files are nonempty; the terminal reload loaded all six tokens.
- Subject exposure/nonzero-gradient counts were 501/501, 501/501, and 498/498;
  color counts were 500/500 for each token. All six initial-to-final deltas were
  nonzero.
- Literal AdamW ordinary-vocabulary drift was recorded, not restored, and not
  thresholded: 49,408 changed rows, mean L2
  `6.765976286260411e-05`, max `0.00021111118257977068`.

## 2026-08-22 — generation and evaluation

The 900-image generation stage was launched from the successful checkpoint at
commit `c8c874d`. Its final status and the independent Grounded-SAM/Qwen stages
are appended after their manifests close; a running process is not counted as
success.

## 2026-08-22 — generation result

- Run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-132122__clevr_subject_color_3x3__generate__c8c874d__42`.
- Manifest status `succeeded`, return code 0.
- Generation manifest: exactly 900 rows: grid 180, subject-only 60,
  color-only 60, transfer 600; 20 distinct seeds, 42–61.
- All 900 referenced images exist and passed Pillow decode, native RGB, and
  512×512 size checks. No image was silently skipped.

## 2026-08-22 — Grounded-SAM compatibility failure and rerun

- First run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-150219__clevr_subject_color_3x3__segment_grounded_sam__c8c874d__42`.
  It wrote 600 explicit failure rows because Transformers 4.48.1 requires
  `box_threshold`, not the newer `threshold` keyword used by the adapter.
- The one-keyword compatibility patch plus regression test passed 11 related
  tests and was pushed as `b059bd5e92cf1994581d8600111d3ed5830dc7d5`.
  The server was clean-fast-forwarded through GitHub only.
- Successful rerun:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-150928__clevr_subject_color_3x3__segment_grounded_sam__b059bd5__42`.
  Manifest succeeded, return code 0; 588 masks were accepted and 12 items were
  explicitly rejected: 6 `no_detection` and 6 `mask_ratio_out_of_range`.

## 2026-08-22 — color metrics

- Run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-151349__clevr_subject_color_3x3__score_color__b059bd5__42`.
- CSV contains 600 transfer rows: 588 scored and 12 `mask_missing`, matching
  the segmentation ledger. No failed segmentation was silently scored.
- Overall 10%/50%/100% means: ΔE 22.0011/31.3093/42.5375; ΔECh
  16.3569/22.6487/30.8198; sRGB angular error
  11.5261/15.8959/20.7788 degrees; hue angular error
  37.0367/52.5351/67.6479 degrees over chromatic valid rows.
- Gray hue is explicitly undefined rather than assigned a fabricated angle.

## 2026-08-22 — Qwen3-VL and CLEVR diagnostics

- Prediction run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-151546__clevr_subject_color_3x3__predict_qwen__b059bd5__42`.
  Manifest succeeded; all 300 non-transfer images produced valid fixed JSON,
  with zero failures.
- Scorer run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-152105__clevr_subject_color_3x3__score_clevr__b059bd5__42`.
  On 180 grid images, shape accuracy was 170/180 = 94.44%, color accuracy
  169/180 = 93.89%, and joint accuracy 169/180 = 93.89%.
- Subject-only contingency: cube→gray 20; cylinder→gray 20;
  sphere→cyan 2 and other 18.
- Color-only contingency (columns cube/sphere/cylinder/other/missing):
  red 1/2/2/15/0; cyan 10/2/1/7/0; gray 3/4/2/11/0.
- These single-axis biases are evidence of residual prompt/output coupling;
  they prevent a categorical “entanglement solved” claim despite strong
  nine-grid joint accuracy.

## 2026-08-22 — final local verification

- All nine test files were run in isolated Python processes to avoid the known
  local Anaconda torch/MKL multi-suite crash: 46 tests passed.
- `compileall` over `src`, `scripts`, and `tests` passed.
- Reproduction/debug JSON files parsed successfully.
- `git diff --check` reported no whitespace errors; only Windows LF/CRLF
  conversion notices were emitted.
