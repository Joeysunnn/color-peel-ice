# Original-image Subject repair: partial, quality unresolved

2026-11-04. The formal source remains the original orange Mailbox photograph
and Scheme A object_003 SAM mask. No historical pink input or mesh was substituted.
Training and transfer remained stopped. No eligible replacement S11 or new
checkpoint was produced by this repair.

## Confirmed integration regression and repair

Historical approved center11 selected the zero-yaw camera from
`mailbox_source_framing72_v1.source_camera`: it preserves source extrinsics,
intrinsics, principal point, distance and bottom crop. The single-image adapter
instead called the complete-mesh centering function, changing composition.
That was an integration error, not a requirement of original-image input or SAM.

Commit `4f16532763d5ddea5c4b89f16c79d8f5d8a60fe1` restores exact saved source-camera
matrices. Recorded review IoU and the first actual training-render IoU must both
reach 0.80; edge contacts require the source mask to reach the same edge within
two pixels. Remaining appearance variants must retain identical mask/normal
buffers. The threshold is an engineering guard, not user geometry acceptance.
The actual previous Mailbox asset was rejected before an output directory was
created, with IoU 0.6618723792556543.

The original/recolor/8-render composition, blue/green RGB factors, metallic
0/1, roughness .15/.85, studio HDRI, resolution, raster settings and SSAO seed
remain aligned with the historical center8 selection. There are still intentional
source/generalization differences: a freshly generated original-image mesh,
SAM instance masks, per-object Qwen category/token initialization, and generic
photo captions instead of the old Mailbox-specific painted-metal captions.
Caption/initializer differences do not cause deformation already visible in the
prepared RGB files. This repair did not alter training or C/M/T.

## Bounded diagnosis and results

| Attempt | New geometry samples | Source-view mask IoU | Outcome |
|---|---:|---:|---|
| Existing seed42 | 0 | 0.6618723793 | Blocked by new guard |
| Fixed 18-view same-mesh diagnostic | 0 | Not an optimization | Elongation visible across front/oblique views |
| Original + identical SAM, seed0 | 1 | 0.7224720861 | Still below guard |
| One camera continuation on unchanged seed0 mesh | 0 | 0.7800232236 | Still below guard; stop |

The seed0 reconstruction used identical source-receipt bytes, cutout and actual
TRELLIS conditioning image. Official source/model hashes, encoder receipt,
precision, sparse algorithm and sampler settings matched the prior seed42 run.
Only the declared model-generation seed changed. One seed was tried, not a search.

The camera continuation used commit `e1e30c4c527a4af94b1dcfd18483c554f5325e71`.
It retained the baseline, objective, bounds and 128px fit resolution, with one
Powell call capped at 1000 evaluations / 10 iterations. It finished successfully
after 418 total objective evaluations, including the baseline. The saved asset
hash matched the pre-run receipt and post-run actual bytes; geometry was not
modified. The final preview remains elongated with a low crest. Improved
projection correspondence is not proof of repaired three-dimensional proportions.

The normal pipeline keeps geometry seed42 and its original camera-fit budget.
It now blocks this known-bad Subject asset. Neither diagnostic seed0 nor camera
continuation became an automatic fallback, and no further attempts were started.

## Mask evidence and causality limit

The actual historical pink reconstruction alpha and current SAM alpha overlap
at IoU **0.9756938818028195** (42,951 versus 42,397 foreground pixels). The earlier
**0.960405668** comparison concerned a different orange-image experiment, not
the historical pink reconstruction. No large missing-subject region or input
stretch was found. Small mask changes may affect generation, but the current
evidence does not establish that SAM caused the deformation. Original versus
pink color, mask details and generated mesh remain confounded in that historical
comparison. There is no evidence that original-image reconstruction is impossible.

## Verification and artifacts

Local related tests: 29 passed. Server Subject/pipeline/training tests: 25 passed;
camera-continuation tests: 4 passed. Both code changes received independent review.
Each GPU run used a fresh GitHub checkout with the complete commit verified,
after checking GPU/process ownership. All repair jobs ended; no training,
transfer, automatic retry, asset deletion or model download occurred.

Server root: `/home/r12user5/Documents/Jiawei/colorpeel-runs/subject_camera_repair_v1/`

- `20261104__same_mesh_probe__da260de`: 18 fixed diagnostic views.
- `20261104__original_seed0__4f16532`: one original-image reconstruction,
  source/model comparison and historical geometry-mask comparison receipts.
- `20261104__seed0_camera_continuation__e1e30c4`: final images, full camera
  trajectory, source receipt, integrity check and blocked Subject eligibility.

Local review bundle:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/subject_original_repair_20261104/`
contains `repair_comparison.png`, `RESULT.json`, plans, execution receipts and
downloaded evidence. Existing failed assets and previously trained checkpoints
were preserved. This is a partial engineering repair, not restored/accepted
original-image Subject quality.
