# CLEVR dataset audit

## Source and evidence state

- Server root: `/home/r12user5/Documents/Jiawei/papers/ICE/datasets/clevr_basic_neutral_stage1_gt`
- Current evidence: tracked adapter executed against the real server root on
  2026-08-22 at commit `c8c874d00318ae7c1df2265c8627787d316a1ce3`.
- Immutable audit artifact:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/train_assets/dataset_audit.json`.
- Audit status: `passed`.
- Training use of GT masks: prohibited; audit/evaluation only.

## Verified inventory

- 48 samples total: `3 shapes × 8 colors × 2 materials`.
- Per sample: 512×512 RGB JPEG, 512×512 binary GT mask, background, scene JSON, and original image copy.
- Verified mask values: exactly `{0, 255}`.
- Verified foreground pixels by shape: cube `13,900`, sphere `14,973`, cylinder `22,868`.
- Verified selected rows: 9; every selected scene label agrees with the locked
  shape/color/metal manifest.
- Verified training staging: exactly nine loader-visible `img.jpg` symbolic
  links. Masks, scenes, backgrounds, and original copies were not linked into
  the sample directories.

The source tree was read-only throughout. SHA-256 values for all 48 source
images, masks, and scene JSON files are stored in the audit artifact.

## Fixed 3×3 metal training grid

| Shape token | Shape | Red (`<c1*>`) | Cyan (`<c2*>`) | Gray (`<c3*>`) |
|---|---|---|---|---|
| `<s1*>` | cube | `003_cube_red_metal` | `013_cube_cyan_metal` | `001_cube_gray_metal` |
| `<s2*>` | sphere | `019_sphere_red_metal` | `029_sphere_cyan_metal` | `017_sphere_gray_metal` |
| `<s3*>` | cylinder | `035_cylinder_red_metal` | `045_cylinder_cyan_metal` | `033_cylinder_gray_metal` |

## Required prompt manifest

Each row must have the official form `a photo of <subject-token> shape in <color-token> color`:

1. `a photo of <s1*> shape in <c1*> color`
2. `a photo of <s1*> shape in <c2*> color`
3. `a photo of <s1*> shape in <c3*> color`
4. `a photo of <s2*> shape in <c1*> color`
5. `a photo of <s2*> shape in <c2*> color`
6. `a photo of <s2*> shape in <c3*> color`
7. `a photo of <s3*> shape in <c1*> color`
8. `a photo of <s3*> shape in <c2*> color`
9. `a photo of <s3*> shape in <c3*> color`

## Staging invariants

- Source dataset is read-only.
- Validate all 48 samples before selecting the nine rows.
- Validate sample ID, scene shape/color/material, RGB size/mode, binary mask size/values, and content hash.
- Refuse missing, duplicate, or extra grid combinations.
- The isolated training tree contains only nine image links in loader-visible image directories.
- Scene JSON, masks, backgrounds, and original copies remain outside loader-visible image directories.
- Generated concepts JSON references the staged links and exact prompts above.
- Store source and staged hashes in the audit output; never mutate source files.

## Reproduction command

See the data audit and staging section of `COMMANDS.md`. The dry-run and real
staging commands both passed before either smoke was launched.
