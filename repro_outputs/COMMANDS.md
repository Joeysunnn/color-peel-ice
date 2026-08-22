# Reproduction commands

Statuses below describe this handoff; a command shown here is not evidence that
the corresponding model run succeeded.

## Local verification — completed

```powershell
cd 'D:\UserFiles\Desktop\sr\color peel'
D:\anaconda3\python.exe -B -m pytest -q
D:\anaconda3\python.exe -B -m unittest discover -s tests -v
git diff --check
```

Observed: `19 passed` in pytest and `10 tests` passed in unittest.

## GitHub push — completed

```powershell
git push --set-upstream origin repro/2026-08-21-colorpeel-clevr
```

`origin` is the user fork; `upstream` is the official repository.

## Server checkout from GitHub — completed for implementation commit

```bash
git clone --branch repro/2026-08-21-colorpeel-clevr --single-branch \
  https://github.com/Joeysunnn/color-peel-ice.git \
  /home/r12user5/Documents/Jiawei/colorpeel
git -C /home/r12user5/Documents/Jiawei/colorpeel status --short
git -C /home/r12user5/Documents/Jiawei/colorpeel rev-parse HEAD
```

Verified clean at `41d752a9d8e8b3a5ab711db90990ab28e4f58000`.

## Environment — not run after rollback

```bash
cd /home/r12user5/Documents/Jiawei/colorpeel
CONDA_BIN=/home/r12user5/miniforge3/bin/conda \
  bash scripts/setup/setup_colorpeel017.sh
```

The previous temporary environment was intentionally removed. Recreate it only
after the optimizer decision is reviewed.

## Data audit and staging — not run with the tracked checkout

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
DATASET=/home/r12user5/Documents/Jiawei/papers/ICE/datasets/clevr_basic_neutral_stage1_gt
STAGING="$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/train_assets"

conda run -n colorpeel017 python \
  src/methods/colorpeel_ice/prepare_clevr_3x3.py \
  --dataset-root "$DATASET" --output-dir "$STAGING" --dry-run
conda run -n colorpeel017 python \
  src/methods/colorpeel_ice/prepare_clevr_3x3.py \
  --dataset-root "$DATASET" --output-dir "$STAGING"
```

## Tracked launcher dry-run — blocked by optimizer decision

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
COMMIT=$(git rev-parse HEAD)
RUN_ID="$(date +%Y%m%d-%H%M%S)__clevr_subject_color_3x3__baseline__${COMMIT:0:7}__42"
python scripts/launch/colorpeel_run.py \
  --config experiments/clevr_subject_color_3x3/configs/baseline.yaml \
  --run-dir "$COLORPEEL_RUN_ROOT/clevr_subject_color_3x3/$RUN_ID" \
  --dry-run
```

Do not remove `--dry-run` until the report's AdamW decision is resolved.

## Generation/scoring entry points — implemented, real run not started

```bash
python scripts/methods/colorpeel_ice/generate.py --help
python scripts/methods/colorpeel_ice/score_clevr_predictions.py --help
python scripts/methods/colorpeel_ice/evaluate_color_metrics.py --help
```
