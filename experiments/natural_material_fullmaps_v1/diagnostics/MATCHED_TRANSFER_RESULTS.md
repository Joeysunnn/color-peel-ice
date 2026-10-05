# Matched material transfer diagnostic (2026-10-04)

This is a visual review of **completed** independent inference runs, not a
training result. The full-resolution PNGs, per-image prompt/seed/hash/safety
ledger, and checkpoint-weight hashes are on research12 under
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_fullmaps_v1/diagnostics/`.

| Run directory | Token checkpoint(s) | Generated images | Safety-filtered images |
| --- | --- | ---: | --- |
| `matched_old_clevr_vs_metal_spoon_20261004` | old CLEVR metal and current metal spoon | 80, plus one discarded base-equality probe | new token: cone plain seed 46; cone photo seed 43 |
| `matched_mailbox_20261004` | current mailbox coating | 40, plus 20 reused base controls and one discarded probe | token: cone plain seed 46; cone photo seed 43 |
| `matched_wood_spoon_20261004` | current wood spoon | 40, plus 20 reused base controls and one discarded probe | token: cone plain seeds 43, 45; cone photo seeds 43, 46; literal: cone photo seed 45 |

All generated images use SD 1.4, PNDM, 100 steps, CFG 3.5, and seeds 42–46.
The noun is `cone` or `mailbox`; each appears as `a {noun}` and
`a photo of a {noun}`. Arms are base, a literal material phrase, and `<M*>`.
The old versus metal-spoon literal control is `polished stainless steel`;
the mailbox and wood controls are `red painted metal` and `light wood`.
The same base prompt, seed and sampling configuration produced **exactly the
same pixels** after loading old metal, metal spoon, mailbox, and wood spoon
token-local K/V checkpoints, as checked by one base probe per new checkpoint.
Base images were then reused in the latter two runs.

## Cone token, conservative per-seed visual read

The labels describe the visible silhouette rather than a model score.
`Pointed` means a recognizably pointed cone-like object. `Frustum/bottle`
means a flat top or narrowed neck. `Other` includes a spout-like object.

| Seed | Old CLEVR, plain | Old CLEVR, photo | Metal spoon, plain | Metal spoon, photo | Mailbox coating, plain/photo | Wood spoon, plain/photo |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | Pointed red cone | Pointed but diamond-like | Frustum/bottle | Cylinder | Cylinder / cylinder | Cylinder / cylinder |
| 43 | Pointed green cone | Pointed green cone | Frustum/bottle | Filtered | Frustum/bottle / filtered | Filtered / filtered |
| 44 | Other, long spout | Other, long spout | Cylinder | Cylinder | Cylinder / cylinder | Cylinder / cylinder |
| 45 | Blue frustum | Green frustum | Cylinder | Cylinder | Cylinder / cylinder | Filtered / cylinder |
| 46 | Pointed red cone | Pointed red cone | Filtered | Cylinder | Filtered / frustum/bottle | Cylinder / filtered |

The old CLEVR token is not perfectly shape-preserving: seeds 44 and 45 are
counterexamples, and seed 42 with `a photo of` is a pointed but unusual
diamond silhouette. Still, it retains a pointed cone-like target much more
often than any current natural-material token at the same prompts and seeds.
The natural-material tokens repeatedly generate a flat-topped cylinder,
frustum, or bottle. The `a photo of` prefix does not restore the cone.

## Mailbox token, visual read

- Old CLEVR token: the five plain and five photo outputs are all mailbox-like,
  although finish, design and framing vary.
- Metal spoon token: the plain outputs remain mostly mailbox-like. Photo
  seed 45 becomes a gray cylinder; other photo seeds range from recognizable
  mailboxes to smooth cabinet or bin-like forms.
- Mailbox-coating token: plain seeds 42, 44 and 46 are recognizable mailboxes;
  43 is a multi-object collage and 45 is a generic white box. In the photo
  arm, seed 44 is a red cylinder and 45 is a plain white box.
- Wood-spoon token: plain seed 42 and 44/46 show at least a mailbox form;
  seed 43 is a collage and 45 a wood block. The photo arm is dominated by
  wood-colored cabinets, boxes, and a cylinder, without reliable mail slots.

The base and literal-material controls also fail the requested noun for
some seed/prompt pairs. These controls must remain visible when judging
transfer; a single image is not an unambiguous material-model failure.

## Interpretation limit

This matched inference **does** show that the old token retains pointed cone
geometry better than the current natural-material tokens under identical
prompts, seeds and inference settings. It does **not** isolate the cause:
material source, renderer, scene, color diversity, caption style, object
scale, and initializer all differ between the old CLEVR and new training.
The old metal is also not a same-material control for mailbox coating or wood.
