# Experiment B: Multi-Color + Multi-Material Subject Learning

This independent SD1.5 Subject study prepares five mailbox colors
(`pink,green,cyan,blue,purple`) x two surface appearances (`metallic,matte`).
The current stage prepares and shows data only;it does not train or generate
model outputs. Existing assets,checkpoints and results are read-only.

The current SD1.5 balanced Subject baseline already uses the same ten RGB
images with explicit `glossy painted metal` / `matte plastic` captions.
Consequently B is a surface-caption ablation against that baseline,not the
first addition of multi-material data. A causal five-metal-only versus
ten-image data ablation would require a separately authorized matched control.

New assets are physical byte-identical copies of the previously reviewed
pairs. The metallic images are also checked against the five images in the
user's original `20260916-163006...bce4364` directory. New directory names and
captions change `red -> pink` and `magenta -> purple`;RGB-LCh hues and pixels
are not recalculated. Source IDs,paths,hashes and the accepted matte review
remain in provenance. The old generated matte reference is a counterfactual,
not measured relighting;internal object details may differ despite matched
binary mask and pixel-identical background.

Each of the ten concepts contains exactly one image,one matching mask and one
caption. This avoids the trainer randomly choosing a material caption for
the wrong image. Captions retain the corrected color supervision:

```text
a photo of <S*> mailbox in purple color with a metallic surface
a photo of <S*> mailbox in purple color with a matte surface
```

## Data preparation only

Run locally modified,committed and pushed code only after the server pulls
the same clean revision. The preparation entrypoint has no training switch.

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
ORIGINAL=$COLORPEEL_RUN_ROOT/20260916-163006__natural_image_subject_color_pilot__subject_recolor_mailbox_training_assets__bce4364__42
REVIEWED=$COLORPEEL_RUN_ROOT/subject_material_composition_v1/20260926-134221__subject_material_composition_v1__mailbox_matte_preview_v2__10a0fb9__42
$PY -B -m experiments.subject_multicolor_multimaterial_b.prepare \
  --original-root "$ORIGINAL" --reviewed-root "$REVIEWED" --output "$NEW_EXTERNAL_RUN"
```

The output contains:

- `dataset/assets/{color}/{material}/`:ten actual RGB/mask copies.
- `dataset/concepts.json`, `training_assets_manifest.jsonl`,source locks and data audit.
- `review/training_pairs.jpg`, `training_captions.csv`, `index.html`:paired image review and every caption.
- `training_plan/configs/{fullrank,lora_r64}.json` and exact baseline parameter comparison.
- `preflight/`:two shared-launcher dry runs;no optimizer updates.
- `inference_plan/step500`, `step750`, `step1000`:fixed prompts and training-image sheets only,no generated inference images.

## Prepared training recipes;not started

The methods inherit their respective frozen SD1.5 Subject recipes:

| Setting | Original | Rank64 |
|---|---|---|
| Update | `<S*>` embedding + full-rank token-local K/V residual | `<S*>` embedding + token-local K/V LoRA |
| Initializer | `mailbox` | `mailbox` |
| Embedding LR | 1e-5 | 1e-5 |
| K/V LR | 1e-5 | 5e-5 |
| Rank / alpha | n/a | 64 / 64 |
| Batch / accumulation / seed | 1 / 1 / 42 | 1 / 1 / 42 |
| Precision / AdamW decay / CAA | no / 0 / 0 | no / 0 / 0 |
| Updates / paired saves | 1000 / 500,750,1000 | 1000 / 500,750,1000 |

The old total doses were2000/3000. The new1000-step dose and snapshot plan
are the user's explicit overrides;all other training arguments are unchanged
apart from the new data path. No core trainer,loss,Subject gate or optimizer
implementation is modified. Ordinary token embeddings must have zero drift.

The `training` entrypoint defaults to preflight only. After separate training
authorization,add `--execute-training`:

```bash
$PY -B -m experiments.subject_multicolor_multimaterial_b.training \
  --dataset "$NEW_EXTERNAL_RUN/dataset" --output "$NEW_TRAINING_OUTPUT" \
  --method fullrank --gpu 2
# The same command with --method lora_r64 prepares the second method.
```

Completed method runs write an audited `training_index.json`. Combine the two
complete selections under `{"git": <same git record>, "runs": [<fullrank audit>,<rank64 audit>]}`
before actual inference. Do not relabel an old checkpoint as a missing step.

## Fixed material transfer plan

Five prompts request metallic,matte,glossy plastic,ceramic or wood surfaces:
`a photo of <S*> mailbox with a {material} surface`.
Each step plans two methods x five materials x seeds42/43 =20 images.
Sampling retains PNDM100,CFG3.5,512px,fp16 and the shared safety checker.
Only the matching Subject adapter and `<S*>` embedding are loaded.

```bash
$PY -B -m experiments.subject_multicolor_multimaterial_b.inference \
  --dataset "$NEW_EXTERNAL_RUN/dataset" --output "$NEW_INFERENCE_ROOT/step500" --step 500
```

That command is a dry run. Actual inference additionally requires both
`--training-index "$COMPLETED_TWO_METHOD_INDEX"` and `--execute-inference`.
Repeat with separate step750/step1000 directories. Subject adapter and token
are exported from the same audited Accelerate snapshot. Result sheets include
all ten actual training images and captions. The previous baseline lacks
every matching step,including750;no matched baseline or disentanglement score
is fabricated. Evaluate subject identity and requested surface separately.
