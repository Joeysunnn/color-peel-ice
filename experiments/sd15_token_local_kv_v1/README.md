# SD1.5 migration: original and rank64 token-local K/V

This separate study preserves the reviewed SD1.4 recipes in `protocol.json`.
Full training configurations change only the base model and run identity.
The migration config locks the SHA256 of canonical parsed protocol JSON, so
Windows/Git line-ending normalization cannot change the protocol identity.
Two-step smoke configurations also change the training dose and save step.
No historical checkpoint, configuration, dataset, or result is overwritten.

| Recipe | Inputs | Method / dose |
|---|---|---|
| `subject_fullrank` | caption-aligned balanced mailbox, 10 images | original full-rank token-local K/V, 2000 steps; snapshots 300/500/700/1000/2000 |
| `subject_lora_r64` | the same balanced 10 images | token-local K/V LoRA rank/alpha64, embedding LR1e-5 / KV LR5e-5, 3000 steps; snapshots 600/1000/2000/3000 |
| `material_ground_reflection_fullrank` | original reviewed 72 controlled metal images | original token-local K/V, 5000 steps |
| `material_{mailbox,metal_spoon,wood_spoon}_lora_r64` | 72 existing full-map renders per material | token-local K/V LoRA rank/alpha64, embedding LR1e-5 / KV LR5e-5, 3000 steps; snapshots 600/1000/2000/3000 |

The material cohorts differ between the historical methods. Compare each
method across SD1.4/SD1.5; this is not a one-variable full-rank-versus-LoRA
experiment. Full-rank here means the token-local residual, not ordinary
all-position Custom Diffusion. Full training has not been started by defining
these configurations. First review the independent native SD1.4/SD1.5 results.

## Model assets

The existing SD1.5 mirror snapshot has the core weights but no safety checker.
`prepare_model.py` creates a new external bundle linking those unchanged
components and the existing SD1.4 checker/feature extractor used for both
native comparison arms. It records all file hashes and refuses an existing
output directory. It never writes into either original model directory and
does not download or install anything.

Use the existing `colorpeel017` Python. Before every operation confirm the same
branch and full commit locally, on GitHub, and on the clean server checkout.
Source changes are local-only, then commit/push, then server fetch/switch and
`pull --ff-only`. Do not merge this experimental branch into research-base.

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$PY -B experiments/sd15_token_local_kv_v1/prepare_model.py
```

After the native comparison review, select a free GPU explicitly. Start with
each method's two-step Subject and Material smoke, then verify adapter and
embedding export/reload in the new base before full training. For example:

```bash
export CUDA_VISIBLE_DEVICES=1
COMMIT7=$(git rev-parse --short=7 HEAD)
STAMP=$(date +%Y%m%d-%H%M%S)
VARIANT=subject_lora_r64_smoke2
RUN=$COLORPEEL_RUN_ROOT/sd15_token_local_kv_v1/${STAMP}__sd15_token_local_kv_v1__${VARIANT}__${COMMIT7}__42
$PY -B scripts/launch/colorpeel_run.py \
  --config experiments/sd15_token_local_kv_v1/configs/${VARIANT}.json \
  --run-dir "$RUN" --dry-run
```

Dry-run creates an immutable run directory. Use a new timestamp/run directory
for actual execution and omit `--dry-run`. All source images, masks, captions,
training-directory membership, model hashes and exact recipe are checked before
launch. `--resume_from_checkpoint` in the inherited trainer is not a working
training-resume implementation; do not use it for this migration.

## Full training kickoff

`run_full.py --batch-dir "$NEW_BATCH_DIRECTORY" --gpus 1 2 3` runs all six
non-smoke configurations, one active training process per GPU, two jobs per
queue. It checks all original input hashes before starting any GPU worker.
A failed job stops its own queue without retrying or deleting artifacts.
The runner waits for child exits to start queued jobs; it does not poll metrics
or generate inference automatically.

On 2026-10-16, `/home` had about 20 GiB free while the preserved checkpoint
schedule alone requires about 23 GiB. Use the independent root-filesystem
output directory `/var/tmp/colorpeel-r12user5/runs` via `COLORPEEL_OUTPUT_ROOT`.
Keep `COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs` for the
original input references. The output override changes only artifact placement;
no training parameter, data path, or old result is altered. The runner requires
40 GiB free before kickoff and before starting each queued job. These new
checkpoints reside under `/var/tmp`, not the historical `/home` output tree;
preserve them there when collecting results.

## Dedicated inference

`evaluate.py` accepts completed training runs from this new study only. It loads
the matching new SD1.5 base and checks full-rank versus rank64 metadata. It can
load a Subject, a Material, or a same-method pair; it rejects old SD1.4 runs and
cross-method or smoke/full mixing. Only final exports are supported at present;
intermediate Accelerate state requires a separate paired export verification.

The prompt manifest is JSONL with `prompt` and integer `seed`; historical
generation manifests can be replayed without carrying old image/checkpoint
paths into the new output. Sampling remains 512px, PNDM100, CFG3.5. For example:

```bash
$PY -B experiments/sd15_token_local_kv_v1/evaluate.py \
  --subject-run "$COMPLETED_SUBJECT_RUN" \
  --prompt-manifest "$FROZEN_TRANSFER_MANIFEST" \
  --output-dir "$NEW_EXTERNAL_INFERENCE_DIRECTORY" --device cuda:0 --dry-run
```

## Matched SD1.4 / SD1.5 comparison

`compare.py` audits the six completed full runs selected in
`inference_protocol.json`. It checks all 19,000 metric rows, the 26 scheduled
adapter/embedding pairs, tensor shapes and finiteness, embedding updates, and
exact equality between each final snapshot and its final export. Paired
inference exports are written into a new output directory; training runs stay
read-only.

Historical SD1.4 manifests, ledgers, provenance and images are hash checked.
The original prompts, seeds and sampling are replayed: PNDM100, CFG3.5, 512px,
float16, a fresh CUDA generator per seed, with the safety checker enabled.
The matrix has 940 original K/V images and 1,620 rank64 LoRA images across 66
jobs, covering the same intermediate steps as the historical evaluations.
The old rank64 Subject comparison's rank4 Material arm is excluded because that
Material is outside these six runs. The Subject-only and literal controls use
the Subject adapter alone; the old unused rank4 adapter has no token gate in
those prompts. Natural-material rank64 composition keeps its own historical
Subject/Material pair and steps.

Run preparation first, inspect its six audit receipts, then launch the queues:

```bash
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
OUT=$COLORPEEL_OUTPUT_ROOT/sd15_token_local_kv_v1/NEW_COMPARISON_DIRECTORY
$PY -B -m experiments.sd15_token_local_kv_v1.compare prepare --output "$OUT"
$PY -B -m experiments.sd15_token_local_kv_v1.compare run --output "$OUT" --gpus 1 2 3
```

Both commands require the same clean branch and commit. Each GPU runs one job
at a time; a failed job stops its queue without retry or overwrite. Completed
jobs automatically produce left-SD1.4/right-SD1.5 comparison sheets. After all
jobs succeed, `index.html` and `summary.json` provide the complete comparison.
Filtered images are marked and counted separately. Finite training losses and
successful image generation do not establish visual quality improvements.
