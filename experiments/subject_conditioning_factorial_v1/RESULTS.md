# Sixteen fresh meshes: Mailbox and wood spoon conditioning factors

2026-11-05. All16 requested reconstructions completed, eight per object,
without failures or retries.64 common-camera geometry views completed.
Results are displayed for user review; no best combination or quality
acceptance is assigned. No training or production-default change occurred.

Producer: `ecac035095b3ea05eac3127dccba5ef44b063ad1`, clean GitHub checkout,
branch `codex/subject-conditioning-factorial-v1`.8 relevant CPU tests passed
locally and on the server. Independent review and scheduler mock checks
covered four concurrent GPU queues, per-GPU serialization and explicit
partial failure without retries. Execution used idle-checked GPUs0-3;
the shared-camera renderer ran after all reconstruction jobs finished.

## Confirmed factor meanings

The user explicitly chose **recolor-only support**: pink is edited inside
SAM or BiRefNet support; outside the selected support RGB retains its original
value. Original RGB is unchanged by the scope factor. Every cell uses the
same frozen uncolored BiRefNet RGBA512 RGB for its object, avoiding the
wood-spoon independent-RGB versus joint-RGBA resizing confound.

Geometry alpha is selected independently: original SAM binary values or the
complete associated BiRefNet soft values. The transparent-RGB-zero receipt
contract is applied after saving pre-alpha RGB. All eight cells share one
crop per object and seed42:

- Mailbox: `[144,253,440,549]`,296x296.
- Wood spoon: `[23,19,487,483]`,464x464.

Thus Mailbox pink/SAM-scope/soft-alpha retains original-colored RGB in extra
BiRefNet support. It is **not** the earlier frozen-old-pink control that had
black RGB there. Mailbox SAM-alpha cells also use296 rather than the old294
crop. These are declared controls, not silent historical replays.

## Completion and reproducibility

Both objects have8 fresh generation calls but6 unique actual conditioning
images and6 unique native files, as expected. For each geometry alpha,
original/SAM-scope and original/BiRef-scope are duplicates. All four duplicate
pairs exactly match actual inputs, configurations, all five native arrays,
voxel metadata and native file SHA256, despite execution on separate GPUs.

The four BiRef-scope/BiRef-alpha baselines (two colors per object) exactly
reproduce their previously recorded native outputs and actual conditioning.
The current study generated these assets again; no old mesh was substituted.
All16 actual conditioning images were checked against preparation, and all
generation configurations/full Git/seed/crop/model revisions were verified.

| Object | Color / recolor support / geometry alpha | Vertices | Faces |
|---|---|---:|---:|
| Mailbox | original / SAM / SAM |736536|1461966|
| Mailbox | original / SAM / BiRef |830457|1664818|
| Mailbox | original / BiRef / SAM |736536|1461966|
| Mailbox | original / BiRef / BiRef |830457|1664818|
| Mailbox | pink / SAM / SAM |870768|1731322|
| Mailbox | pink / SAM / BiRef |916888|1839236|
| Mailbox | pink / BiRef / SAM |1194713|2354418|
| Mailbox | pink / BiRef / BiRef |1033002|2067646|
| Wood spoon | original / SAM / SAM |377292|758920|
| Wood spoon | original / SAM / BiRef |408127|818898|
| Wood spoon | original / BiRef / SAM |377292|758920|
| Wood spoon | original / BiRef / BiRef |408127|818898|
| Wood spoon | pink / SAM / SAM |396437|795250|
| Wood spoon | pink / SAM / BiRef |408780|820056|
| Wood spoon | pink / BiRef / SAM |411049|824758|
| Wood spoon | pink / BiRef / BiRef |416923|836782|

All16 per-asset camera optimizations report nonconvergence under the existing
bounded budget. Their PBR/clay previews remain available and explicitly
marked as separately fitted cameras. Main comparisons instead use one frozen
reference camera and three identical canonical cameras per object, opaque
native triangles and fixed Lambert lighting. No per-mesh alignment,
recentring, scaling, smoothing or geometry edits were applied.64 renders and
their unchanged-geometry hashes passed. Neither face counts nor fitted
silhouette IoU certify geometry quality; alpha targets also differ between
the SAM and BiRef rows.

## Retained artifacts

Server run root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/subject_conditioning_factorial_v1/20261105__mailbox_wood_8each__ecac035__42`.

For each `{object}` in `mailbox`, `wood_spoon` and each
`{color}__{scope}__{alpha}` factor label, the native mesh is:
`{object}/variants/{color}__{scope}__{alpha}/trellis/native_asset.pt`.

- `status.json`:16 jobs, argv/log/GPU mapping and two completed render stages.
- `input_factor_audit.json`: independent pre-alpha RGB and duplicate inputs.
- `numerical_results.json`: all paths, native hashes, counts and fitted metrics.
- `{object}/inputs/`: frozen RGB, per-cell RGB/alpha/condition and locked receipts.
- `{object}/duplicate_original_{sam,biref}.json`: exact duplicate comparisons.
- `{object}/historical_{original,pink}_biref_replay.json`: exact old-baseline comparisons.
- `{object}/geometry_views/`:32 common-camera renders per object and provenance.

Local gallery:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/subject_conditioning_factorial_20261105/index.html`.

Static figures use rows original/SAM, original/BiRef, pink/SAM, pink/BiRef;
columns geometry alpha SAM and BiRef. One shared union crop per object is
used solely for display, with all full-canvas images retained. The wood-spoon
bowl close-up uses the same fixed source512 ROI `(200,20,304,185)` for every
cell. An additional Mailbox yaw+90 matrix displays the complete projected
shapes: the old source-view camera itself clips some meshes at the512 canvas
boundary. The yaw matrix likewise uses one union crop for all eight cases.
HTML provides full-canvas reference/canonical clay, normals, masks,
separately fitted PBR/clay and actual model conditioning.

Downloaded evidence archive SHA256:
`3f17b28be9227a15cd51c392ad4dfd971b6227545621d5de1795bb24ddbe18d3`.
Native assets remain on the server; their hashes were calculated there.
