# Constructed Mailbox counterfactual dataset, 2026-10-27

180 synthetic subject–material images were constructed from the same complete
native Mailbox geometry. The dataset includes180 image/mask/caption concepts,
30 matched six-material blocks, and210 metallic-flip/adjacent-roughness pairs.
No additional3D sample, training, or monitoring was started.

## Execution and location

- Branch: `codex/trellis2-counterfactual-dataset-v1`.
- Execution commit: `0d7fce4b5cd43bff86258845170936a841d7029e`.
- Server run:
  `/var/tmp/colorpeel-r12user5/runs/subject_trellis_dataset_v1/20261027-192223__material180__0d7fce4__42`.
- Same path plus `.log`: captured execution log.
- Same path plus `.delivery.tar.gz`: complete delivery excluding compiled
  Triton cache. Archive SHA256:
  `25eefdffdef502dcb8e62e5c36808c8c44e60dc1deefa5c48e8eca3926f75be4`.
- Rendering, packaging and full technical verification took36.843s after
  runtime/light initialization. Asset preparation and transfer are excluded.

The code was developed and checked locally, committed/pushed, then cloned
into a new server checkout at the exact execution commit. It reused the
isolated TRELLIS runtime on physical GPU1; no packages were installed.

## Dataset factors

Five uniform Base Colors (pink, green, cyan, blue, purple), two metallic
values(0/1), three roughness values(.15/.50/.85), three near-source camera
orientations and two official HDRIs(studio/courtyard) form the complete grid.
Each of30 view/color/light blocks contains exactly the same six M/R states.
Caption text and scalar controls are recorded for every image.

Every render uses the full919,180-vertex/1,836,844-face native geometry,
opaque texture-free material, the same geometric normals and seed42 SSAO.
The predicted Base Color and M/R are replaced rather than recolored, removing
their baked appearance. Geometric relief remains; texture-only patterns are
discarded. Cameras are centered to show the full generated asset, with yaw
offsets−.25/0/+.25 from the estimated original orientation, not a return to
the photograph's cropped framing.

## Verification

All checks passed, with180 unique combinations, 30 blocks and210 edges:

- Rendered Base Color/M/R/Alpha G-buffers match the stated constants.
- Geometry masks and normal buffers are identical across conditions per view.
- RGB, binary512 masks, RGBA cutouts and captions are correctly paired.
- Training RGB retains raw rendered foreground; exterior is exactlyRGB127.
  Transparent hidden RGB is zero. No photograph pixels are pasted back.
- Every concept directory contains exactly one image, with one caption.
- All image/map hashes match; there are no NaNs, clipping of geometry, missing
  combinations or byte-identical foregrounds along the210 intervention edges.
- Native asset, camera receipt, HDRIs, protocol and clean source state passed
  before/after locks. Native geometry hashes are unchanged.
- Five meaningful CPU tests passed, including wrong mask/material and
  duplicate-factorial rejection, and exact single-factor pair structure.

Foreground RGB mean absolute differences(on0–255 display values):

| Intervention | Pairs | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Metallic0→1, same R | 90 | 25.752 | 41.165 | 81.218 |
| Adjacent R, same M | 120 | 2.638 | 13.583 | 68.197 |

These differences confirm image response, not perceptual material accuracy or
learned disentanglement. Actual180 RGB images and captions are available in
`review/index.html`, six contact sheets and `review/captions.csv`.
Manual inspection of both center-view light grids shows the same mailbox
relief and silhouette, sharp highlights at low roughness, broader/duller
responses at high roughness, and strong metallic/dielectric appearance changes.
This supports the intended candidate review; it is not a human acceptance
decision or a training result.

Dataset manifest SHA256:
`49825606688d019563d16f148392cf34a7931c1ccbf48d866bbb7850e2df6bfc`.
Concepts SHA256:
`d77f11b75ae4d3969ff9f943dcd9096db9ba9367db325fd081ec01af7abc3f6c`.
`verification.json` and `repro_outputs/status.json` preserve the complete
technical evidence; candidates remain pending user review and untrained.

## Interpretation and next use

This dataset supplies controlled material interventions while keeping a
single generated subject mesh fixed. It is not evidence that the mesh exactly
matches the real Mailbox, that texture-only identity was retained, or that
Subject training will successfully disentangle material. It covers M/R
reflectance, without labeling wood, cloth or other unimplemented materials.

`dataset/concepts.json` matches the base trainer's image/mask/caption structure.
It was not passed to training. The existing MARBLE35/14 staging must not be
used unchanged, and the base trainer does not enforce the candidate manifest's
eligibility flags. All samples share one geometry; a random image split would
not establish independent-source generalization.
