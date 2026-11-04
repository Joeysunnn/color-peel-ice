# Historical reproduction succeeded; current-SAM pink remains below the guard

2026-11-04. The two authorized reconstructions finished. The historical native
asset and accepted camera review were reproduced exactly. Recoloring the current
SAM cutout to pink improved source-view correspondence, but did not produce an
eligible replacement S11. No training or transfer ran, and no production default
changed. Visual quality has not been accepted by the user.

## Results

| Input and reconstruction | Seed | Source-view mask IoU | Result |
|---|---:|---:|---|
| Existing original orange + current SAM | 42 | 0.6618723792556543 | Baseline; not rerun here |
| Exact historical pink RGBA replay | 42 | 0.814327485380117 | Exact historical native asset and review |
| Original-to-pink RGB + unchanged current SAM | 42 | 0.7804528027073918 | Improved; blocked by unchanged 0.80 guard |

The new pink camera fit converged in 270 objective evaluations within the original
350-evaluation budget. This is distinct from the previous orange **seed0** camera
continuation, whose IoU was 0.7800232236 after 418 evaluations. No continuation or
additional seed was used here.

In the source-camera clay previews, current pink is visibly less elongated than
current orange. Its crest silhouette, body proportions and projection still
differ from the source photograph. This observation concerns the rendered source
view; it does not establish correct hidden geometry or separately quantify camera
and geometry error. IoU is a two-dimensional overlap guard, not quality approval.

## Stage 1: exact historical reproduction

Generation checkout: `681c77b4bcadb0561a30b302ff8908f157febb81`.
Comparison checkout: `9cc989c5a923ef298f8886ddb12f39e64f1fe3a4`.

Reference native asset:
`/var/tmp/colorpeel-r12user5/runs/subject_trellis_counterfactual_v1/20261027-110302__pink_asset__e0ca373__42`.

Accepted six-parameter camera-review reference:
`/var/tmp/colorpeel-r12user5/runs/subject_trellis_counterfactual_v1/20261027-111312__camera_review__b4a76cd__42`.

- Frozen RGBA input and actual 298x298 conditioning pixels matched exactly.
- Official source revision, encoder receipt, model hashes, sampler configuration,
  precision, sparse algorithm and latent normalization matched.
- All five native arrays (`vertices`, `faces`, `coords`, `attrs`, `origin`) matched
  exactly in original order, dtype and shape, without alignment or tolerance.
  Voxel metadata/layout also matched. There are 919,180 vertices and 1,836,844 faces.
- Camera extrinsics/intrinsics and IoU matched the later accepted review. All eight
  saved PNGs (source RGB, PBR, clay, base color, metallic, roughness, normal and
  fitted mask) matched byte-for-byte.
- Only secondary GLB export was skipped. Native generation and review completed.

Thus the prior successful result is reproducible with the recorded historical
input and configuration. It is not necessary to substitute the old mesh to obtain
that result. This does not make the historical manually repaired alpha an
automatic mask-generation method.

## Stage 2: current SAM, pink conditioning only

Generation and preparation checkout: `9cc989c5a923ef298f8886ddb12f39e64f1fe3a4`.

Parent receipt:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/single_image_four_branch_v1/20261104__mailbox_A_full__da8e990__42/prepared/objects/object_003/input/source_receipt.json`.

The input remained rooted in the original orange photo. The experiment applied
the historical deterministic Lab/LCh hue=0 recoloring with soft-alpha RGB blending
to the foreground. It did not apply either historical manual mask repair.
Only the TRELLIS conditioning cutout changed; the source photograph, original
photo-anchor path and C/M/T sources were preserved.

The binary SAM alpha was exactly unchanged (42,397 foreground pixels; mask SHA256
`44c8a935359f52c97f3cc788168d0103fc8482295be1f9531efff537a2052d0c`).
There were zero RGB changes outside the mask; 922 edge pixels used soft RGB
blending. The actual conditioning crop stayed 294x294. Original image512 SHA256:
`644a77fe664bc2d9b0c94ada86259ce5a810a530580e754f1b23a40a82906be5`.

Model/configuration/seed matched the orange baseline. The new native mesh contains
1,134,277 vertices and 2,249,226 faces. Calling the existing Subject renderer
rejected review IoU 0.7804528027073918 before creating a Subject dataset directory.
No photographic anchors, 8-render set, S11 or checkpoints were created in this run.

Holding the current mask and generation settings fixed while changing foreground
RGB changed the result. Therefore mask change alone is not an adequate explanation
for the failure. However, the fitted projection also uses a weak RGB term, so the
IoU gain cannot be attributed entirely to geometry. The remaining historical-vs-
current difference still includes the historical manually repaired alpha, crop
and recolor-boundary details. This experiment does not isolate their contributions
or establish pink as generally optimal for arbitrary inputs.

## Verification and retained outputs

The code change received independent review. All 36 related tests passed locally
and in the clean server checkout. Full Git revisions and GPU/process ownership
were checked before execution. No model downloads, source copying, asset deletion,
other-task interruption or production input/default changes occurred.

Server root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/subject_pink_input_repro_v1/`

- `20261104__historical_replay__681c77b__42/`: `trellis/native_asset.pt`, review,
  generation log, `native_comparison.json` and `review_comparison.json`.
- `20261104__current_sam_pink__9cc989c__42/`: `pink_inputs/source_receipt.json`,
  `trellis/native_asset.pt`, actual condition, review, generation log,
  `controlled_comparison.json` and `subject_eligibility.json`.

Local evidence and comparison figure:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/pink_input_repro_20261104/`.

Both jobs have ended. The experiment stops here with historical reproduction
confirmed and current-SAM pink quality unresolved. The normal pipeline still
uses original orange conditioning, seed42, its original camera budget and the
existing 0.80 Subject guard.
