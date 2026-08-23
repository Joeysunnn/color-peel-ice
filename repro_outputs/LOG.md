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

## 2026-08-22 — human review and diagnosis-first decision

- The report-01 checkpoint was designated the frozen comparison anchor. The
  source training run remains
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130816__clevr_subject_color_3x3__baseline__c8c874d__42`
  at `c8c874d00318ae7c1df2265c8627787d316a1ce3`. No baseline artifact was
  overwritten or resumed by this documentation work.
- Human semantic review is the primary follow-up evidence. The available review
  has no per-seed ledger and is therefore `observed`: grid mostly correct with
  roughly one or two black images; subject-only cube/cylinder gray and sphere
  mostly gray with a small cyan minority; color-only red mostly red with no
  consistent CLEVR-shape leakage, gray faithful, cyan less stable and sometimes
  cube-like; red/gray transfer strong and cyan transfer poor.
- Qwen remains secondary. Its sphere-only color result (`other` 18/20) conflicts
  with the human observation of mostly gray spheres. No evaluator error rate is
  computed because the human review is not yet itemized.
- Interpretation decision: current evidence supports paired-template success
  plus single-axis context dependence. Cyan-to-cube is a diagnosis target, not
  confirmed token entanglement. Black outputs, segmentation failures, and color
  error are not independently treated as subject-color leakage.

## 2026-08-22 — initializer evidence and conditional ablation

- Direct server check used the cached SD 1.4 tokenizer through
  `AutoTokenizer(..., local_files_only=True)` in `colorpeel017`.
- Verified IDs: `cube [11353]`, `sphere [6987]`, `cylinder [22092]`,
  `red [736]`, `cyan [1470, 550]`, `gray [7048]`, `aqua [18613]`,
  `teal [22821]`, and `turquoise [19899]`.
- Consequence: the frozen baseline's configured `cyan` is multi-piece. Future
  initializer validation rejects multi-piece candidates rather than silently
  taking one piece. The baseline itself is not altered.
- Conditional config:
  `experiments/clevr_subject_color_3x3/configs/train_cyan_initializer.yaml`,
  variant `cyan_initializer_ablation`. At that pre-diagnostic snapshot it used
  a runtime candidate placeholder; the post-review config now locks the single
  selected candidate `turquoise` directly.
- No candidate is selected. No initializer-ablation dry-run, smoke, full
  training, generation, or evaluation is claimed.

## 2026-08-22 — pending diagnostic and later-stage boundaries

- `diagnostics_v1` is specified as three ordered black-image stages: FP16 with
  the default SafetyChecker and recorded NSFW flag; FP16 with the checker
  disabled only after explicit acknowledgement; then checker-disabled FP32
  with finite learned-weight and generated-pixel audits. Each stage uses a new
  output directory and only continues IDs that remain black.
- The cyan diagnostic specifies 540 images across ten nouns, seeds 42–44, two
  prompt families, 300 trained-K/V rows, and 240 vanilla-SD rows. Trained rows
  include learned `<c2*>` and literal `cyan`/`aqua`/`teal`/`turquoise`; vanilla
  rows include the four literal words. A separate utility randomizes all 540
  images into single-image blind-review rows and writes a condition key.
- Entry points and protocols existing in the worktree are pre-run evidence only.
  No diagnostic image, completed review CSV, or follow-up metric is recorded.
- `clevr_multiview_protocol.json` defines the conditional held-out protocol;
  real rendering and all nine `multiview_fold_{a|b|c}_seed{42|43|44}` runs
  remain pending. The
  protocol planner must record a missing renderer as blocked rather than
  fabricating views.
- A factor-aware loss and natural multi-object evaluation remain conditional.
  No approved config, implementation result, run path, or output is claimed.

## 2026-08-22 — diagnosis-first local verification

- Dry-run manifests contained exactly 540 cyan diagnostic images, 75 subject
  diagnostic images, and 180 multiview render requests. The multiview planner
  correctly returned `blocked` without a compatible renderer and created no
  images.
- Full collection initially aborted inside NumPy after pytest collected the
  PyTorch training-audit module. The color suite alone and a direct
  PyTorch-plus-color calculation both passed; this isolated the failure to the
  local Windows Anaconda OpenMP runtime combination, not the color equations.
- With `KMP_DUPLICATE_LIB_OK=TRUE` scoped only to the local pytest process, all
  73 tests passed in 18.38 seconds. Compileall, 8 JSON files, 9 YAML files, and
  `git diff --check` also passed. The environment flag is not part of any
  server run command.

## 2026-08-22 — server deployment and ordered black-image result

- Pushed `ca3d313c4d081bcdec5fda6979b05c4fde3415c0` to
  `origin/repro/2026-08-22-colorpeel-diagnostics`. The server checkout was
  clean before deployment and now tracks the same branch and commit; no code
  was copied or edited directly on the server.
- The server `colorpeel017` environment passed all 73 tests in 5.40 seconds and
  compileall passed.
- Runtime root:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-191800__diagnostics_v1__ca3d313__42`.
- Stage 1 kept FP16 and the default SafetyChecker. It selected all 19 exact-black
  source IDs; 19/19 reruns recorded `nsfw_content_detected=true` and remained
  black.
- Stage 2 kept FP16 and changed only SafetyChecker disablement with explicit
  acknowledgement. All 19 outputs were finite and nonblack, with 19 successful
  status rows. Therefore Stage 3 FP32 was not run.
- Source-aware color evaluation wrote 600 rows: 588 successful and 12 existing
  mask failures. GT-mask median references were red `[83,35,35]`, cyan
  `[38,91,91]`, and gray `[58,58,58]`. Mean nominal/source DeltaE at the 50%
  pixel fraction was red 16.5341/30.3232, cyan 66.0792/42.5543, and gray
  12.0731/12.9418. Source-aware scoring explains part, but not all, of cyan's
  transfer deficit.
- PID 2806571 was launched for the checker-disabled 540-image cyan diagnostic,
  followed by the 75-image subject diagnostic and blinded-review packet. At
  this evidence snapshot the job was running; no candidate or training result
  was selected.

## 2026-08-22 — completed diagnostics and initializer decision

- The diagnostic process completed. The server contains 540 cyan PNGs and 540
  `ok` status rows, 75 subject PNGs and 75 `ok` status rows, plus 540-row review
  and condition-key CSVs.
- Human folder-level review found that every learned subject token combined
  correctly with literal red, cyan, and gray. Subject-only gray is therefore
  treated as a base-model/default-completion effect rather than evidence that
  the subject token cannot combine with color.
- Under trained K/V, learned `<c2*>` was the only consistently poor cyan
  condition. The four literal words worked under both trained K/V and vanilla
  SD; trained K/V was qualitatively somewhat more stable against variegation.
- The user selected `turquoise` as the best trained-K/V single-token
  initializer. This review was qualitative and inspected named condition
  folders rather than completing the randomized blind-review ledger; no win
  rate is inferred.
- The next run changes only `<c2*>` initialization from the historical
  first-piece-of-`cyan` behavior to token ID 19899 (`turquoise`). New 2-step and
  9-step smoke configs precede a fresh 1500-step run. CAA, AdamW, training mask,
  prompts, data, seed, K/V scope, steps, and all other initializers stay fixed.
- The first server dry-run invocation appended an unsupported `__preflight`
  run-ID segment. The launcher rejected it before creating a manifest or
  starting training. The command record was corrected to the required
  `TIMESTAMP__study__variant__commit7__seed` form.

## 2026-08-22 — turquoise smoke and full training evidence

- GitHub/server training commit: `0959d1ed5ce99553dfa107987efd84b6313a972c`.
- The corrected dry-run succeeded and recorded initializer list
  `cube+sphere+cylinder+red+turquoise+gray`.
- The turquoise 2-step run
  `20260822-233300__clevr_subject_color_3x3__smoke2-turquoise-first-two__0959d1e__42`
  passed its validator. It recorded two finite metric rows, the exact two
  expected token pairs, nonzero updates for all seen tokens, and seven nonempty
  final weight artifacts.
- The turquoise 9-step run
  `20260822-233340__clevr_subject_color_3x3__smoke9-turquoise-full-grid__0959d1e__42`
  passed its validator. All nine pairs appeared once; all six tokens had three
  exposures and real updates.
- Full run
  `20260822-233433__clevr_subject_color_3x3__cyan_initializer_ablation__0959d1e__42`
  succeeded with return code 0, 1500/1500 finite metric rows, and a complete
  `checkpoint-1000`.
- Final exposure/nonzero-gradient counts were subject `501/501`, `501/501`,
  `498/498` and color `500/500` for each token. `<c2*>` delta was `0.0231073`.
  All seven final weight files were nonempty and the training script completed
  its final reload path. Ordinary-vocabulary drift covered 49,408 rows with
  mean/max L2 `6.76598e-05`/`2.21111e-04`, recorded without enforcement.
- Training success does not establish transfer improvement. The next evidence
  is a fresh matched 900-image run and human review.

## 2026-08-23 — matched turquoise inference and human gate

- Matched run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-235330__clevr_subject_color_3x3__generate__388dc56__42`.
- Launcher manifest: `succeeded`, return code 0, commit `388dc56`.
- The generation manifest contains 900 rows; every row records
  `safety_checker_disabled=true` and `safety_risk_acknowledged=true`.
- Read-only image audit found 900 RGB 512×512 PNGs, zero exact-black images,
  and zero decode/mode/size failures.
- The user manually inspected the new inference folders and reported broad
  improvement after the `turquoise` change, with outputs now mostly meeting
  requirements. This qualitative decision closes the single-view gate and
  authorizes the next multiview held-out stage. No numerical human accuracy or
  claim of general disentanglement is inferred.
- The generation manifest has 900 unique IDs and SHA-256
  `60e8597629199f59e856a8e246056247c32bb7fb9cf4d360ff0c26aa01624db1`.

## 2026-08-23 — multiview preflight

- The locked fold structure passed review: each of the nine cells is held out
  once, every axis value retains two training partners, and each fold has 96
  training views plus matched audit records.
- Preflight found that fold configs still inherited the frozen baseline's
  multi-piece `cyan` initializer. A dedicated turquoise multiview base config
  now prevents that deterministic startup failure without rewriting baseline.
- Realization validation now rejects nonempty output directories, repeated
  images presented as multiple views, absent camera/light/background variation,
  protocol-identity drift, and contaminated image-only staging.
- No compliant renderer currently exists in the three local repositories.
  The closest modern Blender script is untracked in the CLEVER checkout and
  has fixed camera/light/background metadata, so it is read-only reference
  material rather than an executable dependency.
- Existing data object coordinates and renderer code support scale 1.3;
  renderer defaults support 512 Cycles samples and fixed object rotation.
  Official CLEVR defines camera jitter 0.5 and per-light jitter 1.0. It does
  not define a neutral-background variation range, so rendering remains
  blocked on that explicit scientific parameter rather than inventing one.
- The complete local regression suite passed 77 tests in 70.93 seconds with
  the previously documented process-local OpenMP compatibility flag. Compile,
  JSON/YAML parsing, and `git diff --check` also passed.
