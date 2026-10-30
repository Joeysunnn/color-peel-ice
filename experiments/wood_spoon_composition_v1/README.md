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
