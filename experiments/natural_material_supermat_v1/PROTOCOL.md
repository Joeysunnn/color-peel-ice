# Natural mailbox material extraction and calibration (v1)

## Scope and gate

This experiment extracts a **visible painted-surface finish** from the original
natural mailbox photo. Its first representation is two scalar Principled BSDF
parameters: roughness and metallic. SuperMat albedo is retained for audit but
its hue never enters the canonical material parameters. The inferred metallic
value describes the visible surface model; the photo cannot establish the
physical substrate underneath the paint.

Do not train `<M*>`, change existing Subject/Color runs, or generate the 96-cell
dataset from this protocol. The dataset stage needs explicit human acceptance
of the calibration sheet and round-trip report.

## Source and mask

- Original photo: `assets/source_mailbox.jpg` (orange mailbox, 512×512), copied
  byte-for-byte from the 010_D1GT_81_130 natural sample.
- Original object mask: `assets/object_mask.png` from the same natural sample.
- `configs/mailbox_body_region.json` defines a deliberately small area of the
  orange curved lid. It excludes the black slot, trim, hardware, silhouette,
  and brick background. `material_pipeline.py prepare` intersects its polygon
  with the original object mask and erodes the object boundary by 5 px. The
  material region is a preliminary visual annotation; inspect
  `assets/region_overlay_preliminary.png` before interpreting its statistics.
- SuperMat receives the whole masked object as RGBA, not the small measurement
  region alone. The measurement region is used only for spatial aggregation.

## Extraction

Use official [SuperMat](https://github.com/hyj542682306/SuperMat) single-image
inference with its `supermat.pth` checkpoint and
`sd2-community/stable-diffusion-2-1` base. Record official commit, environment,
checkpoint and base-model IDs, CLI arguments, input/output hashes, and seed.
Input is 512×512 RGBA with zero alpha outside the object. The official loader
composites transparent pixels against neutral gray. Run with `--save-orm` to
retain the packed map, but read the separate `roughness.png` and `metallic.png`
files for aggregation. Their grayscale values are divided by 255. Use the
median of pixels within the material region; also report MAD, IQR, 5th and
95th percentiles. A wide or bimodal distribution is a reason to revise the
region, not to silently accept the median.

The source photo, object mask, measurement region, RGBA input, gray composite,
maps, canonical JSON, and their hashes live in a new run directory under
`colorpeel-runs/natural_material_supermat_v1/`. The source Albedo map remains
an audit artifact only.

## Renderer calibration

The renderer is independent of the legacy CLEVR Rubber/Metal node groups. It
uses Blender Cycles and a Principled BSDF with fixed neutral gray Base Color,
extracted roughness and metallic, one sphere, and three neutral lighting
conditions: `soft_front`, `side_directional`, `top_environment`. Geometry,
camera, light transforms, render seed, samples, device, color management,
shader inputs, source hashes and git commit must be recorded per image.

Review the sheet for a plausible matte/glossy response, metallic appearance,
and moving highlights across lights. The target is fixed underlying material
parameters. Rendered pixel RGB is not expected to match the source photo.

## Round trip and acceptance

For each calibration render, use its object mask to make a 512×512 RGBA input,
rerun the same SuperMat checkpoint, then aggregate an eroded interior mask.
Report `R_hat - R*` and `metallic_hat - metallic*` per light plus map spreads.
This is a cross-renderer consistency diagnostic, not physical ground truth:
SuperMat and Blender have different rendering assumptions, and a single image
cannot uniquely determine reflectance under unknown illumination. Keep the
raw maps and per-light deltas even if the check fails.

Human review must accept: the measurement region, plausible 3-light material
sheet, no obvious exposure/shader defect, and round-trip deltas interpreted
with their spreads. A failure pauses the 96-cell stage for a documented
single-variable revision; it does not trigger token training.

## Dataset stage, after approval only

The later matrix is four shapes (sphere, cube, cylinder, cone) × four externally
controlled base colors (red, blue, green, yellow) × three lights × two views
(front-ish, 45°), at most 96 images. Every cell uses the same accepted
roughness and metallic values. Save each image, object mask, prompt/label,
input RGB, geometry/light/view configuration, seed, source material JSON hash,
renderer metadata, file hashes and git commit. A cone needs a new recorded
procedural geometry definition because the current CLEVR asset set has none.

## References

- SuperMat paper: https://arxiv.org/abs/2411.17515
- Official code and inference instructions: https://github.com/hyj542682306/SuperMat
- Official checkpoint: https://huggingface.co/oyiya/SuperMat
