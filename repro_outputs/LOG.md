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
- Preflight commit `50b745a1da8c1418a360eac5b9180a4c7e0b36dc`
  was pushed to the fork. The clean server checkout fast-forwarded from
  `388dc56` using only `git pull --ff-only`; no server code file was edited.
  The server `colorpeel017` environment then passed all 77 tests in 6.34
  seconds and the checkout remained clean.

## 2026-08-23 — fixed-neutral 180-view rendering

- Blender 4.2.11 archive SHA-256 matched
  `7f084fd57f1351bcae3434fc5450643547e4ad3d69cd93d4dd14a784203ee2ec`;
  it was installed only under `/home/r12user5/Documents/Jiawei/tools/`.
- Final smoke:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260823-105000__clevr_subject_color_3x3__multiview-render-smoke__53a3a0e__42`.
  It used one CUDA-visible Tesla V100, produced RGB 512×512 and strict
  complementary binary masks, with mask ratio `0.0501289` and no edge contact.
- Full run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260823-110000__clevr_subject_color_3x3__multiview-render__53a3a0e__42`.
  Renderer status was `succeeded`: 180 realization rows, 180 `ok` status rows,
  zero failures, and no `.partial` directory.
- `realize` validated all 180 views and produced `prepared_human_review/`:
  180 review rows, a 45-image contact sheet, three folds × 96 image-only
  training files, and nine configs. All configs use the turquoise initializer
  and status `pending_human_review`; GT masks are absent from training staging.
- Runtime exposed only operational validator issues: Blender Cycles arguments
  must occur after `--`, and Blender stores transforms as float32. The adapter
  now rejects non-CUDA Cycles CLI values and uses a measured `1e-6` transform
  tolerance (maximum observed rounding error `2.383989e-7`). No renderer,
  ColorPeel loss, optimizer, prompt, or data value changed.
- Final validator/workflow commit: `de279caef13f4e9580825d4e1c5347fd14f5faab`.
  Local and server suites each passed 90 tests. No Fold training was started.

## 2026-08-23 — versioned orbit renderer v2 smoke

- v2 was added without modifying the v1 profile/protocol/schema. The sole
  scientific change is camera sampling: object-centered orbit offsets of
  ±18° azimuth, ±10° elevation, and ±5% distance, with explicit `-Z`/`Y-up`
  look-at. Fixed background, object settings, light draws, seeds, splits,
  ColorPeel training code, CAA, AdamW, and masks are unchanged.
- v1 historical fingerprints remain
  `246cb06778a74f994311c3e1e3a8a4aa973ce7a308d3cd2732dcdcc021bf8529`
  for the profile and
  `97162f88528794f03e553dee70bfacc83b20646587a5ed9a7617f2c23f818c2d`
  for the canonical requests.
- Local and server unittest discovery each passed 62 tests after the v2
  implementation. The server checkout was updated only by GitHub
  `git pull --ff-only` and ended clean at
  `bb7593042413eba55caa347102dffe8106f7bf7b`.
- Two runtime-only faults stopped safely before final images: the base camera's
  active `Track To` constraint, then a stale dependency-graph location read.
  v2 records/mutes base constraints and refreshes the view layer before its
  explicit look-at. These fixes do not touch v1.
- Runtime root:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260823-140000__clevr_subject_color_3x3__multiview-render-v2-smoke__28a51bb__420000`.
  Five cube views and one cylinder view succeeded under real Blender 4.2.11,
  Cycles CUDA, one visible V100 on GPU 3, and 512 samples.
- All six outputs passed orbit metadata, object-center, fixed-background,
  RGB/mask, complement, area, edge, and hash checks. Cube foreground counts
  were 12,488–14,271 pixels; the cylinder count was 21,222. The five cube RGB
  hashes were unique; alignment and Y-up cosines were within floating-point
  tolerance of 1.0.
- Quick visual inspection showed clear cube face-proportion changes, stable
  framing/background, and a complete cylinder with visible top ellipse. No
  full v2 180-view render, realization, contact sheet, or Fold training ran.

## 2026-08-23 — full orbit renderer v2 realization

- After the user accepted the v2 smoke, the clean server checkout at
  `f7bc52dffa32345ee09cc25804d67866f68f49de` executed a fresh canonical run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260823-200500__clevr_subject_color_3x3__multiview-render-v2__f7bc52d__420000`.
- Blender 4.2.11/Cycles CUDA used GPU 3. The full command had neither
  `--limit` nor `--resume`; it succeeded with 180/180 realization rows,
  180/180 `ok` status rows, zero partial outputs, and 20 unique RGB hashes in
  each of nine cells.
- `realize` validated all 180 views. It generated a 180-row human-review CSV,
  a 45-image contact sheet, three image-only folds of 96 JPEGs each, and nine
  turquoise configs. Each concept directory contains exactly 16 JPEGs and no
  other files; all configs remain `pending_human_review`.
- The contact sheet is 1000x2016 and has SHA-256
  `0a674ce117d7d34f87c0750fdf48da1fef3aec03336f95574529aaa577ae655cb`.
  Evidence inspection found visibly changing cube faces and cylinder
  top/side proportions, expected sphere highlight/shadow movement, fixed
  neutral backgrounds, and no obvious black or clipped sampled image.
- No Fold smoke, 1500-step Fold training, factor-aware loss run, or natural
  multi-object run was started. The next action remains the user's review of
  the v2 contact sheet and review ledger.

## 2026-08-23 — multiview v2 Fold training kickoff

- The user manually accepted the full v2 contact sheet and explicitly released
  the Fold training gate. No new method or loss was authorized.
- All nine derived configs passed launcher dry-run: folds A/B/C × seeds
  42/43/44, 96 image-only training views per fold, and 1500 steps per run.
- A detached sequential queue started on GPU 3 at commit `9cf1446`. It verifies
  the pinned commit and clean worktree before every run and stops on the first
  failure. Campaign root:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260823-203240__multiview_v2_fold_training_campaign__9cf1446`.
- Fold A / seed 42 was the first active run. Startup verification observed 141
  finite metric rows, all six modifier tokens exposed, and an active V100
  process. The run later crossed step 1000 and wrote `checkpoint-1000` with
  model, optimizer, scheduler, random-state, and custom-checkpoint artifacts.
  No completed run or held-out result is claimed yet.
- Fold A / seed 42 then completed 1500/1500 with return code 0. All metric rows
  were sequential and finite; six-token exposure/nonzero-gradient counts were
  480–512 with positive embedding deltas. Final Custom Diffusion weights, six
  token files, embedding audit, and post-save reload were present. The queue
  advanced automatically to Fold A / seed 43. No held-out result is claimed.

## 2026-08-24 — nine Fold runs complete and held-out campaign prepared

- The sequential GPU-3 queue ended with `CAMPAIGN_SUCCESS` at
  `2026-08-23T21:49:51+08:00`. All nine Fold/seed runs (A/B/C × 42/43/44)
  returned 0 and contain exactly 1500 finite, sequential metric rows.
- Every run has six distinct expected modifier-token audit rows with positive
  embedding deltas, checkpoint-1000, final K/V plus six token files, and
  post-save reload evidence. Ordinary-vocabulary drift remains record-only
  under literal official AdamW behavior.
- Added a versioned complete-bundle held-out protocol: nine checkpoints × nine
  complete subject/color prompts × seeds 42–61 = 1620 images (1080 seen, 540
  held-out). Cyan remains the target label; `turquoise` remains only the
  single-token initializer.
- The generation adapter disables the previously confirmed false-positive
  SafetyChecker only with paired acknowledgement, binds each row to parent
  training/config/model/protocol hashes, and supports provenance-aware resume.
- Nine successful generation runs must pass an immutable merge step before
  human review or Qwen. The merge validates every 512×512 RGB/hash/status,
  writes a randomized 1620-row human-review ledger and nine 180-image contact
  sheets. Qwen is incremental/resumable and remains secondary to human review.
- No held-out image, metric, disentanglement claim, factor-aware loss, or
  natural multi-object run is claimed by this preparation update.

## 2026-08-25 — three-attribute implementation

- Created an isolated 3x3x2 material protocol with paired metal/rubber orbit
  views, 360 render requests, 288-image full-grid staging, and 192-image Fold
  staging.
- Added eight-token smoke auditing, strict initializer/token cardinality, and
  preserved the literal official AdamW embedding decay behavior.
- Added full-grid and held-out generation manifests, provenance-aware resume,
  three-key Qwen parsing, per-axis/joint scoring, and three separate
  intervention tables. Human review remains primary and no single
  entanglement score exists.
- This code update stops before training. A real Blender metal/rubber smoke,
  v2 metal hash match, full 360 render, and renderer human gate are required.

## 2026-08-25 — renderer v3 paired smoke stopped at byte-hash gate

- Server checkout `16ccdbe` passed 129 tests. SD1.4 locally cached tokenizer
  confirmed `metal [4044]` and `rubber [11331]` as single tokens.
- The paired real Blender smoke produced two valid CUDA/V100 images at seed
  420000 with identical camera/light metadata and pixel-identical binary masks.
- Byte SHA did not reproduce the accepted v2 image. A fresh unchanged-v2
  control also differed from accepted v2 by 108 channel values of magnitude 1;
  v3 metal differed by only 30 values of magnitude 1.
- Per the locked plan, the 360-view render was not started and the validator
  was not weakened. A protocol decision on pixel-level equivalence is required.

## 2026-08-26 — decoded-pixel equivalence gate approved

- The user explicitly approved replacing cross-run byte equality with the
  measured decoded-pixel gate. RGB must have maximum absolute channel
  difference at most 1 and mean absolute channel difference at most 0.001;
  object/background masks remain decoded-pixel exact.
- Raw accepted-v2 and v3 SHA-256 values are retained in a 180-row audit and are
  no longer treated as proof of cross-run pixel inequality.
- Renderer v3 profile, 360 requests, seeds, camera, lights, materials,
  background, Cycles settings, ColorPeel training and all historical v1/v2
  fingerprints remain unchanged. No training is authorized by this decision.

## 2026-08-28 — full-run gate v2 approved

- The fresh renderer run completed 360/360 with no failures, `resume=false`
  and `limit=null`. The v1 realization gate stopped before staging because 40
  of 180 metal images had a sparse decoded-channel maximum above 1.
- Full read-only audit found 23,093 changed values among 141,557,760 channels
  (0.0163%). Per-image maximum changed fraction was 0.000831604; maximum mean
  difference was 0.000930786; the observed maximum channel difference was 5.
- All 180 accepted-v2 object/background masks and all 180 metal/rubber paired
  seed, camera, light and decoded masks were exact. Every one of 18 cells had
  20 unique RGB hashes.
- The user approved `decoded_pixel_equivalence_v2`: mean and changed-channel
  fraction must each be <=0.001, maximum difference is recorded only, and mask
  equality remains strict. Existing 360 renderer artifacts will be reused; no
  renderer or training setting changes and no training is authorized.

## 2026-08-28 — material realization completed at human gate

- Commit `774241a` passed 133 server tests after a clean GitHub-only
  fast-forward. The existing 360-view renderer output was reused without a
  Blender invocation or resume.
- All 180 metal references passed gate v2. Maximum mean was
  `0.0009307861328125`, maximum changed-channel fraction was
  `0.00083160400390625`, maximum single-channel difference was 5 (record-only),
  and raw RGB SHA matches were 0/180. Both decoded masks matched 180/180.
- Realization produced 360 rows, an 180-row equivalence audit, 288 full-grid
  JPEGs and 192 JPEGs per Fold, with no non-JPEG staging files. It also wrote a
  360-row review CSV, one 18-cell sheet and nine paired sheets.
- `human_gate_decision.json` remains `pending_human_review` and
  `training_authorized=false`. No two-step smoke, coverage smoke, 1500-step
  training, generation or Qwen stage was started.

## 2026-08-28 — material renderer human gate accepted

- The user manually accepted the material contact and paired sheets and asked
  to enter the next stage.
- The released action is the locked sequential two-step startup smoke followed
  by the independent 18-step coverage smoke on GPU 3. Each run must pass its
  own evidence validator before the full-grid baseline may start.
- This decision does not authorize Fold training and does not establish any
  material-token or disentanglement result.

## 2026-08-28 — material dual smoke passed

- Launcher dry-runs and real executions completed for the independent 2-step
  paired-material and 18-step full-grid coverage configs at commit `6fdd889`.
- Both manifests returned 0 and both `src.train.training_audit` results passed
  with finite losses, exact expected prompt/token coverage, nonempty K/V and
  eight-token artifacts, and successful reload.
- The 18-step run observed subject/color exposure 6 and material exposure 9;
  every token had real gradients on every exposure and positive delta.
- Literal official AdamW drift affected all 49,408 ordinary rows at very small
  magnitude and remained record-only. No optimizer or training semantics were
  changed.

## 2026-08-29 — material full-grid baseline reached human gate

- The locked seed-42 full-grid run completed 1500/1500 finite metric rows,
  checkpoint-1000, final Custom Diffusion weights, all eight token artifacts,
  embedding audit, and successful reload/launcher exit.
- All eight learned tokens had nonzero gradients on every exposure and positive
  initial-to-final deltas. Literal official AdamW drift of ordinary vocabulary
  rows remained record-only.
- Full-bundle inference completed 360/360 valid 512x512 RGB images for 18
  prompts and seeds 42-61. All generation status rows were `ok` and all image
  hashes matched.
- An SSH reset stopped the foreground generator after 76 files. Those files
  passed manifest, RGB, model/protocol fingerprint and SHA-256 recovery checks;
  the exact run then resumed under `nohup` and completed without overwriting
  them. No scientific setting changed.
- The first bundle attempt exposed a direct-script `src` import bug. Commit
  `d48ffa8` fixed only the entrypoint path and added a regression test; the
  server passed 134 tests before the replacement bundle succeeded.
- The final bundle contains a 360-row randomized review CSV, a 40-row
  red-sphere regression CSV and nine same-seed material pair sheets. Its gate
  remains pending and Fold training is not authorized.

## 2026-08-30 — material full-grid human gate accepted

- Human review found metal and rubber stably distinguishable and confirmed that
  material replacement preserves the requested shape and color.
- Known appearance artifacts remain: some metal objects have a strong glossy
  upper region and weakly glossy lower region, while some rubber objects show
  noticeable highlights but remain recognizable as rubber.
- The gate is recorded as `passed_with_known_material_appearance_artifacts`.
  This releases only the locked nine-run Fold campaign on GPU 3; it does not
  authorize a renderer, data, loss, optimizer, or prompt change.
- Held-out review must track the two artifact classes separately. No claim of
  illumination-invariant material learning or solved disentanglement is made.

## 2026-08-30 — material held-out Fold training completed

- The detached serial GPU-3 campaign completed all nine Fold/seed runs at
  commit `dcb465c` and ended with `CAMPAIGN_SUCCESS`.
- Independent aggregation verified 13,500/13,500 optimization steps, exactly
  1500 sequential finite metric rows per run, checkpoint-1000, final Custom
  Diffusion weights, all eight token files and successful final reload.
- All eight tokens had nonzero gradients on every exposure and positive
  initial-to-final deltas in every run. Ordinary-vocabulary AdamW drift remained
  record-only.
- Final artifacts are used per run; there is no metric-based model selection.
  This completion releases the locked 3240-image generation campaign but does
  not itself establish held-out generalization or disentanglement.
