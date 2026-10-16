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

The current definitions and CPU tests do not prove SD1.5 training compatibility.
No claim of migrated training results is made until the server smokes and the
matching full runs have their own successful evidence.
