# Wood spoon Subject + Material / Texture

Inference only, using the user's selected existing checkpoints:

| Branch | Source | Step | Initializer |
| --- | --- | --- | --- |
| Subject | TRELLIS blue/green72, wood_spoon,600-step run | 600 | spoon |
| Material | wood_spoon natural Material initializer ablation | 1000 | material |
| Texture | wood_spoon natural Texture16 | 1000 | texture |

The Subject source is
`/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons600_v1/20261028-140359__blue_green72_both600__bb84ce1__42/wood_spoon/step600`.
Material comes from `natural_material_initializer_v1/20261019-095320__wood_material_init_material__9178429__42`;
Texture comes from `natural_texture_material_v1/20261018-190309__natural_tm_training__9077221__42`.
Both latter paths are below `/var/tmp/colorpeel-r12user5/runs`.

For each of original/fullrank token-local K/V and rank64/alpha64 token-local K/V:

```text
a photo of <S*> spoon made of <M*>
a photo of <S*> spoon with <T*> texture
```

Use the same seeds30–60 inclusive across all four conditions:124 images total.
PNDM100,CFG3.5,512px,fp16,safety checker, independent device generator per image.
Always load both token embeddings and their corresponding two adapters, in
Subject/attribute order. Each branch's token and adapter use its own same-step
snapshot; S600 is deliberately combined with M/T1000. S+M never loads Texture;
S+T never loads Material. The inherited second slot's internal name `material`
also carries T, which is explicitly recorded as Texture in provenance.

`source.prepare` verifies the exact historical sources and existing paired
exports. Subject audits run in the original cleanbb84ce1 checkout; Material
and Texture audits validate only the requested wood concepts with their
recorded training revisions, actual configs and same-step tensor equality.
Do not call the old SD1.5 `verify_run` for T-rank64: that migration-only helper
does not support Texture metadata. No source metadata is relabeled.

All source data, masks, tokens, checkpoints and old outputs remain read-only.
Deploy through local edits/tests/commit/push, then server clone/pull into a
new clean checkout. Keep its HEAD fixed during inference. Choose available
physical GPUs immediately before launch; do not disturb another user's jobs.

```sh
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.wood_spoon_composition_v1.run \
  --output "$NEW_EXTERNAL_ROOT" --gpus 0 2 3 --execute
```

Omitting execute creates a separate immutable source-audit/preflight directory.
Before dispatch, all four31-image inference plans are checked. The finite
queue launches one process per GPU and stops an affected queue on failure,
without retry or retraining. Verify initial images/processes, then stop
checking; no monitor or automatic notification is created.

All four conditions stay below the same new experiment directory. `index.html`
links per-condition results and, after completion, shows the two methods for
both compositions side by side for each seed. Individual pages show all actual
S72 and M8/T16 training RGB references/captions. Source audits, old/new Git
revisions, prompts, hash/safety ledgers and sampler are recorded separately.
Successful generation alone is not a subject/attribute disentanglement verdict.

## Texture500 rerun (2026-10-30)

The screenshot-selected Texture source is the same natural Texture training run,
but its **step500** paired export is used instead of step1000. Training total was
1000; both its adapter and `<T*>` embedding must come from snapshot500. Subject
remains the user's wood_spoon step600. Match fullrank with fullrank and rank64
with rank64, using the same prompt and seeds30–60: **62 images**.

```sh
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.wood_spoon_composition_v1.run \
  --output "$NEW_EXTERNAL_ROOT" --gpus 2 3 \
  --texture-only --texture-step 500 --execute
```

This mode audits only the two S600 and two T500 pairs, dispatches only S+T,
and keeps both methods together under a fresh experiment root. The previous
S+M and S+T1000 outputs stay unchanged. `index.html` compares both methods per
seed and links their actual S72/T16 training references. No training or monitor.

## Subject200/400 with fixed Texture500 (2026-10-30)

Test the other saved Subject snapshots from the **same600-update** wood spoon
run:200 and400. Keep Texture500 and all prompts/sampling/seeds fixed. Use both
methods, matching each Subject adapter and embedding at its selected step with
the corresponding method's Texture500 adapter and embedding. No training.

```sh
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.wood_spoon_composition_v1.run \
  --output "$NEW_EXTERNAL_ROOT" --gpus 0 2 3 \
  --texture-only --texture-step 500 --subject-steps 200 400 --execute
```

All four31-image conditions (124 images) share one fresh experiment directory:
`step200_texture/{fullrank,lora_r64}` and `step400_texture/{fullrank,lora_r64}`.
The root page compares four columns by seed. Both source audits and all four
preflights complete before GPU dispatch. The Subject training total remains600;
only the selected snapshot changes. Preserve previous S600+T500 and all other
results. Choose GPUs from a fresh occupancy check, verify startup, then stop
checking as requested. The defaults still select S600.

## Texture750 with Subject200/400/600 (2026-10-31)

Use the existing natural Texture **step750** paired adapter and embedding,
matching methods with Subject200,400,600 from the same600-update wood spoon run.
The Texture training total remains1000; no training or re-exporting is needed.
Keep the same prompt and seeds30–60:3 snapshots ×2 methods ×31 seeds =186 images.

```sh
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.wood_spoon_composition_v1.run \
  --output "$NEW_EXTERNAL_ROOT" --gpus 0 2 3 \
  --texture-only --texture-step 750 --subject-steps 200 400 600 --execute
```

Use a fresh output root containing all six conditions under
`step{200,400,600}_texture/{fullrank,lora_r64}`. The root page compares six columns
for each seed after completion. Audit all sources and preflight all six jobs
before dispatch, check first outputs, then disconnect without monitoring.
Previous Texture500 and other results stay unchanged. Check GPU occupancy
immediately before launch. Default selection remains S600/T1000.

## Collect72 and180 Subject tests with Texture500 (2026-10-31)

Copy the six completed S72 step200/400/600 + T500 groups into one new experiment,
preserving every original file byte and historical provenance. Validate all31
seeds, PNG hashes, source locks, training references, prompt and sampler first;
write an additional reuse receipt. These186 images are never regenerated.

Audit the original wood180 Subject step750/1000 pairs from the completed
continuation. Reuse the old fullrank training at its original revision and the
completed rank64 continuation, matching each method with Texture500. Generate
four31-image groups (124 new images), with actual180 training RGB references.
No training or checkpoint export is performed.

```sh
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.wood_spoon_composition_v1.collect_t500 \
  --output "$NEW_EXTERNAL_ROOT" --gpus 0 2 3 --execute
```

All310 images share a fresh root with `s72_step{200,400,600}/{fullrank,lora_r64}`
and `s180_step{750,1000}/{fullrank,lora_r64}`. After completion `index.html`
shows ten columns per seed. The copied methods keep their original provenance;
new methods record both source training revisions. The two dataset sizes also
have different total training steps, so this is not a controlled dose comparison.
Keep seed30–60, PNDM100, CFG3.5,512px,fp16 and checker. Complete all audits and
four preflights before dispatch; verify startup, then stop checking.

## Literal wood spoon + Texture500 (2026-10-31)

Use `a photo of a wood spoon with <T*> texture`, with the literal words
`wood spoon` and **only** the Texture500 embedding and token-local K/V adapter.
No Subject token, adapter, source audit or Subject references are loaded.
Both original/fullrank and rank64/alpha64 use seeds30–60:62 images.
PNDM100/CFG3.5/512px/fp16/checker and fresh generator per image stay fixed.

```sh
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.wood_spoon_composition_v1.run \
  --output "$NEW_EXTERNAL_ROOT" --gpus 2 3 \
  --texture-step 500 --literal-subject --execute
```

Only T500 sources are audited. The pipeline loads a single token/adapter and
applies a single-position mask for `<T*>`; ordinary wood/spoon words retain
their base embeddings. The literal baseline has no learned subject identity.
Keep both methods in a new root; pages display T16 actual RGB references and
same-seed comparisons. Preserve all old outputs. Verify startup, then disconnect
without monitoring. No training or re-exporting.

## Eight seed42 controls (2026-10-31)

Fix rank64/alpha64, Texture500 and Subject72 step600. Compare four subject
conditions: literal mug, spoon, wood spoon, and learned `<S*> spoon`. Each has
two arms: no Texture, or `with <T*> texture`. The no-T arm removes its embedding
and K/V adapter together with the prompt token; learned Subject keeps S600 in
both arms. No-T ordinary words use the same unadapted common loader as the
historical mug baseline. All use fresh seed42, PNDM100/CFG3.5/512/fp16/checker.

Reuse the old mug native/T500 pair, literal wood spoon/T500 and S600/T500
images after exact prompt, source selection, sampler, safety and PNG hash
checks. Copy historical provenance and ledgers into each case's reuse evidence;
keep every source file unchanged. Generate only the missing spoon no-T/T500,
wood spoon no-T and S600 no-T images. All eight share one fresh output root.

```sh
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.wood_spoon_composition_v1.control_run \
  --output "$NEW_EXTERNAL_ROOT" --gpus 2 3 --execute
```

The root page has four subject rows and two no-T/T500 columns, recording which
images were copied or generated. Case pages retain prompts, source records,
safety/hash and actual training references for loaded concepts. All existing
weights/results are read-only. Audit sources and preflight all missing cases
before dispatch; verify startup, then disconnect without monitoring or retry.
