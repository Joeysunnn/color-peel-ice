# Archived emission color in the C+M study

## What the earlier color run learned

The archived orange source is the approved `D1GT:81/130.png` emission render:
Cycles emission strength 1.0, three CLEVR shapes by three views, nine images.
The captions were `a photo of {subject} shape in <C*> color`; no object masks
or subject token entered color training. The orange token was initialized from
`orange`. The run used SD 1.4, seed 42, 100 steps, effective learning rate
1e-5, AdamW weight decay 0.01, and full Custom Diffusion cross-attention K/V.
The archived K/V and token weights are pinned by SHA-256 in
`protocols/archived_emission_color_comparison_v1.json`.

The old `cos_weight: 0.2` was active. With one learned token, the old trainer
paired `<C*>` with the next ordinary prompt position in its attention cosine
calculation; this is not a color/material disentanglement penalty. The current
trainer intentionally rejects that one-token setting. The old `hflip: true`
flag did not flip images because the dataset transform was commented out.
The old 100/100 transfer result counts successfully written images at CFG 6;
it is not a 100/100 color-quality score. Human review already noted occasional
literal orange fruit.

## Controlled reuse

`compare_archived_emission_color.py` uses the archived full-K/V `<C*>` as-is
and adds the selected frozen `<M*>` K/V residual at material-token positions.
It compares this with the current token-local `<C*>` plus the same `<M*>`.
Both branches use the same cube, sphere, and held-out mug prompts, seeds
42–44, 100 steps, CFG 6, safety checker, and source checkpoint hashes. The
paired condition is inference composition; no checkpoint is retrained.

The archived color K/V acts on every text position, including the material
word and token. This faithfully retains the old color mechanism but creates
a possible interaction with the material adapter. Compare color-only with
paired outputs within each branch. Keep the earlier 36-image CFG 3.5 run as
a separate baseline because its prompt wording also differs.

## Improvements to test after visual review

1. Evaluate orange body coverage, object identity, metallic appearance,
   background leakage, and fruit artifacts on predeclared valid shapes;
   generation success alone does not measure these properties.
2. Match prompt wording and CFG in every comparison. Treat old full-K/V
   versus token-local K/V and old adjacent-word CAA as separate changes;
   the existing two training runs changed both at once.
3. If a new token-local color run is needed, test object-masked loss and
   longer training as separate ablations. The nine emission images have
   limited shape and material diversity; keep a natural noun such as mug
   held out to measure transfer.
4. Fix or remove the inactive `hflip` flag in a separately versioned data
   change. Set embedding weight decay to zero or restore ordinary rows if
   strict vocabulary isolation is required: both short color runs changed
   ordinary embedding rows slightly under AdamW weight decay 0.01.
