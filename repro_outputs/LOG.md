# Reproduction log

## 2026-08-21 — repository and protocol audit

- Read the full official ColorPeel code at baseline commit `021f5c7` and the
  ColorPeel/ICE papers.
- Locked the CLEVR metal grid, six tokens, prompts, official parameters, and
  evaluation seeds in the approved plan.
- Direct server inspection established the reported 48-sample inventory.
- Found the official final-modifier gradient off-by-one defect and the separate
  unresolved AdamW weight-decay behavior.

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
- Local verification: pytest `19 passed`; unittest `10 passed`; all method CLI
  help commands, baseline config parsing, and Bash syntax checks succeeded.
- Created implementation commit `41d752a9d8e8b3a5ab711db90990ab28e4f58000`
  and pushed the reproduction branch to the fork.

## 2026-08-22 — server checkout from GitHub

- Cloned the fork branch directly into
  `/home/r12user5/Documents/Jiawei/colorpeel/`.
- Verified branch `repro/2026-08-21-colorpeel-clevr`, clean status, remote URL,
  baseline config presence, and HEAD `41d752a9d8e8b3a5ab711db90990ab28e4f58000`.
- No environment, model, data staging, training, or evaluation command was run.

## Open decision

Literal official AdamW updates zero-gradient embedding rows through decoupled
weight decay. The current patch fixes only the sixth-token boundary, as
approved. Training remains blocked until the user selects literal behavior or
an explicit non-modifier restoration patch.
