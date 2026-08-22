# ColorPeel-ICE project and server layout

## Source of truth

The GitHub fork is the only source of code. Development happens locally, then reviewed commits are pushed to that fork. The server directory `/home/r12user5/Documents/Jiawei/colorpeel/` is populated and updated only with GitHub `clone`, `fetch`, and `pull`/fast-forward operations at an explicit commit.

Do not deploy code to the server with IDE auto-upload, SCP, rsync, copied archives, or direct server edits. A server run must record the fork URL, branch, full commit hash, and clean-worktree status before execution.

The upstream repository and pinned baseline commit remain provenance anchors:

- upstream: `https://github.com/moatifbutt/color-peel`
- baseline commit: `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- working branch convention: `repro/YYYY-MM-DD-short-task`

The GitHub fork is `https://github.com/Joeysunnn/color-peel-ice.git`; the
official repository remains the read-only `upstream` remote.

## Server roots

```text
/home/r12user5/Documents/Jiawei/colorpeel/              GitHub-derived code checkout only
$COLORPEEL_DATA_ROOT/<dataset>/                          immutable inputs
$COLORPEEL_CACHE_ROOT/                                   reusable model/package caches
$COLORPEEL_RUN_ROOT/<study_slug>/<run_id>/               immutable run artifacts
```

Recommended shell contract:

```bash
export COLORPEEL_DATA_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-data
export COLORPEEL_CACHE_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-cache
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
```

All three roots must resolve outside `/home/r12user5/Documents/Jiawei/colorpeel/`. The existing CLEVR input at `/home/r12user5/Documents/Jiawei/papers/ICE/datasets/clevr_basic_neutral_stage1_gt` is treated as immutable external input; it is not a code checkout or run directory.

## GitHub-only server update contract

Initial checkout:

```bash
git clone https://github.com/Joeysunnn/color-peel-ice.git /home/r12user5/Documents/Jiawei/colorpeel
git -C /home/r12user5/Documents/Jiawei/colorpeel fetch --all --prune
git -C /home/r12user5/Documents/Jiawei/colorpeel checkout <explicit-branch-or-commit>
git -C /home/r12user5/Documents/Jiawei/colorpeel status --short
git -C /home/r12user5/Documents/Jiawei/colorpeel rev-parse HEAD
```

Later updates:

```bash
git -C /home/r12user5/Documents/Jiawei/colorpeel fetch origin
git -C /home/r12user5/Documents/Jiawei/colorpeel checkout <explicit-branch>
git -C /home/r12user5/Documents/Jiawei/colorpeel pull --ff-only origin <explicit-branch>
git -C /home/r12user5/Documents/Jiawei/colorpeel status --short
git -C /home/r12user5/Documents/Jiawei/colorpeel rev-parse HEAD
```

Replace placeholders before use. A dirty checkout or a commit mismatch blocks a run.

## Run contract

Each job is declared by a tracked YAML file under `experiments/<study_slug>/configs/` and receives a unique immutable directory under `$COLORPEEL_RUN_ROOT`. Use a run ID such as:

```text
YYYYMMDD-HHMMSS__clevr_subject_color_3x3__baseline__<commit7>__42
```

Launch a tracked config only from a clean checkout:

```bash
RUN_ID=YYYYMMDD-HHMMSS__clevr_subject_color_3x3__baseline__<commit7>__42
python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/baseline.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID" \
  --dry-run
```

Review the dry-run provenance, delete `--dry-run` only after the unresolved
optimizer decision is recorded, and never reuse a run directory.

Before model execution, validate configuration, manifest completeness, roots, clean Git state, and the AdamW embedding-freeze decision. The run directory should preserve:

```text
config.yaml
command.sh
environment.txt
manifest.json
logs/
checkpoints/
inference/
evaluation/
figures/
```

No run has been launched under this contract. Its current status is `pending`.
