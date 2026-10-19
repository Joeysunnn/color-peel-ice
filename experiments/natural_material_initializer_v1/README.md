# Natural Material initializer ablation

Retrain the selected source's two natural Material recipes with
the selected initializer (`material` by default) instead of `metal`. All other training arguments,
the eight RGB images/masks/captions,and extracted R/M are identical to the
completed natural training. Each source retains its own data and raw R/M.
Run metadata changes to isolate this experiment.
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
  --output "$NEW_EXTERNAL_RUN" --source-id wood_spoon --gpus 1 2 3
```

`--source-id` accepts `wood_spoon` (default), `mailbox`, or `metal_spoon`.
`--initializer` defaults to `material`; `wood` is allowed only for `wood_spoon`.
For the wood-initialized Material ablation, retain the same environment and
source paths,select a fresh output under the same experiment parent,and run:

```bash
$PY -B -m experiments.natural_material_initializer_v1.run \
  --source-training "$BASE/20261018-190309__natural_tm_training__9077221__42" \
  --source-transfer "$BASE/20261018-213417__ordinary_transfer__9d70d7a__42" \
  --output "$NEW_EXTERNAL_RUN" --source-id wood_spoon --initializer wood --gpus 2
```

One GPU serializes both training jobs and all three transfer steps.
Three distinct GPUs retain the original two training workers and three
transfer workers. This changes scheduling only;training and sampling args
remain fixed. The current tokenizer must encode the initializer as one token.
The two learned contact-sheet columns are labelled with their actual old and
new initializer words. Existing `learned_material_init` image IDs denote the
new Material-branch arm;the actual word is recorded in the plan,provenance
and training references,including `wood` for this experiment.

For the two remaining sources, use the same source paths/environment with:

```bash
$PY -B -m experiments.natural_material_initializer_v1.remaining_sources \
  --source-training "$BASE/20261018-190309__natural_tm_training__9077221__42" \
  --source-transfer "$BASE/20261018-213417__ordinary_transfer__9d70d7a__42" \
  --output "$NEW_EXTERNAL_BATCH" --gpus 1 2 3
```

Set the fresh batch output below
`/var/tmp/colorpeel-r12user5/runs/natural_material_initializer_v1/`.
Mailbox training and transfer finish before metal_spoon starts, so the sources
do not compete for GPUs. Both are scheduled immediately and run unattended.
The batch stops on a failed source. It writes its own status, command log and
index; each source has independent `mailbox/` or `metal_spoon/` subdirectories
with `step500/`, `step750/`, `step1000/`, exports and actual training references.
The two-source batch totals four1000-step runs,twelve paired exports,
96 new images,288 reused controls,and24 contact sheets.

Use a new output directory. The runner audits the original training/exported
tensors,dataset,and baseline transfer provenance before writing an exact
training-argument comparison. It verifies that the selected initializer is one CLIP token,
performs two launcher dry runs,then trains from the frozen SD1.5 base on the
selected GPU(s). There is no checkpoint resume. The rank64 trainer allowlist
accepts `<M*>` with `material` or `wood` only in token-local mode.

Successful training automatically exports six paired500/750/1000 snapshots
under `exports/`,then runs three transfer steps on the selected GPU(s).
The same chair/mug/vase/sofa prompts,seeds42/43,PNDM100,CFG3.5,512px,fp16
and safety checker are reused. Each step generates16 new-token images and
copies48 hash-verified old native/literal/metal-initialized controls into its
own `step500/`, `step750/` or `step1000/` directory. Old images are never edited.

Each step produces64 image files and four contact sheets. Four columns show
native SD1.5,literal material,old `metal` initialization,and the selected new
initialization. Both learned columns share the same eight actual training
images displayed above the grid;run/checkpoint provenance is labelled.
Literal controls retain the frozen mapping:mailbox=`painted metal`,
metal_spoon=`polished metal`,wood_spoon=`wood`.
Totals:48 new images,144 reused controls,12 sheets. No Texture token/adapter
or texture prompt is loaded. There is no automatic disentanglement score.

`train_outputs/` records commands,config differences,logs and status.
Root `execution_verified` requires both training audits,six paired exports,
three successful complete transfer ledgers,and all12 contact sheets.
