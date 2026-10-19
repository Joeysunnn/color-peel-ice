# Wood Material initializer ablation

Retrain only the two natural wood_spoon Material recipes with
`initializer_token=material` instead of `metal`. All other training arguments,
the eight RGB images/masks/captions,and extracted R/M are identical to the
completed natural training. Run metadata changes to isolate this experiment.
The Texture branch and existing results are read-only.

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
BASE=/var/tmp/colorpeel-r12user5/runs/natural_texture_material_v1
$PY -B -m experiments.natural_material_initializer_v1.run \
  --source-training "$BASE/20261018-190309__natural_tm_training__9077221__42" \
  --source-transfer "$BASE/20261018-213417__ordinary_transfer__9d70d7a__42" \
  --output "$NEW_EXTERNAL_RUN" --gpus 1 2 3
```

Use a new output directory. The runner audits the original training/exported
tensors,dataset,and baseline transfer provenance before writing an exact
training-argument comparison. It verifies that `material` is one CLIP token,
performs two launcher dry runs,then trains from the frozen SD1.5 base on the
first two GPUs. There is no checkpoint resume. The rank64 trainer allowlist
additionally accepts `<M*>` with `material` only in token-local mode.

Successful training automatically exports six paired500/750/1000 snapshots
under `exports/`,then starts three transfer workers on the selected GPUs.
The same chair/mug/vase/sofa prompts,seeds42/43,PNDM100,CFG3.5,512px,fp16
and safety checker are reused. Each step generates16 new-token images and
copies48 hash-verified old native/literal/metal-initialized controls into its
own `step500/`, `step750/` or `step1000/` directory. Old images are never edited.

Each step produces64 image files and four contact sheets. Four columns show
native SD1.5,literal `wood`,old `metal` initialization,and new `material`
initialization. Both learned columns share the same eight actual training
images displayed above the grid;run/checkpoint provenance is labelled.
Totals:48 new images,144 reused controls,12 sheets. No Texture token/adapter
or texture prompt is loaded. There is no automatic disentanglement score.

`train_outputs/` records commands,config differences,logs and status.
Root `execution_verified` requires both training audits,six paired exports,
three successful complete transfer ledgers,and all12 contact sheets.
