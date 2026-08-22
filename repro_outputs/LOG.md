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
