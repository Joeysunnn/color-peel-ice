# Multiview orbit renderer v2

## Scope and sole scientific change

`multiview_render_v2` is a versioned renderer variant. It does not modify or
replace `multiview_render_v1`.

The sole scientific change is the camera sampler: v1 adds XYZ translation
jitter while preserving the base camera rotation; v2 extracts the base
radius, azimuth, and elevation relative to the realized object's center, then
applies deterministic orbit offsets of ±18 degrees azimuth, ±10 degrees
elevation, and ±5 percent distance. The camera local `-Z` axis is recomputed to
look at the object center with local `+Y` as up. The base scene's tracked
camera constraints are recorded and muted before this explicit v2 look-at is
applied; v1 constraint behavior remains untouched.

Object pose/scale/material, neutral world and ground values, white-light
settings and jitter range, Cycles seed, resolution, samples, render assets,
view splits, held-out folds, ColorPeel training settings, CAA, AdamW, and mask
boundaries are unchanged. Because the three v2 camera draws replace the three
v1 camera-translation draws in the same RNG positions, the light offsets are
identical between v1 and v2 for a given render seed.

## Runtime gate

Run real Blender 4.2.11/Cycles CUDA smoke renders before any complete v2
realization. Synthetic unit fixtures are validator tests only and are not
render evidence. Do not start Fold training from v2 until all 180 views pass
realization and a new human review explicitly releases the gate.

## Runtime smoke evidence

Commit `bb7593042413eba55caa347102dffe8106f7bf7b` passed the real Blender
runtime gate on GPU 3. The isolated run root is:

`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260823-140000__clevr_subject_color_3x3__multiview-render-v2-smoke__28a51bb__420000`

Five canonical cube views and one cylinder view were rendered with Blender
4.2.11, Cycles CUDA on one V100, and 512 samples. The cylinder smoke used a
permutation of the same 180 canonical request rows so `--limit 1` could select
`035_cylinder_red_metal:view_00`; it is smoke-only and is not eligible for
realization. All six outputs passed RGB/mask decode, strict object/background
mask complement, area/edge checks, fixed-background comparison, deterministic
orbit metadata validation, object-center target comparison, and artifact
hashing. The five cube RGB SHA-256 values are distinct.

The first runtime attempt exposed the base scene's active `Track To`
constraint and stopped before rendering. v2 now records and mutes base camera
constraints before applying its explicit look-at. A second pre-render attempt
exposed a stale dependency-graph location read; v2 now refreshes the view layer
and computes direction from the requested orbit location. Both are v2-only
runtime corrections; failed roots are retained as evidence and v1 is unchanged.

Visual smoke inspection found the cube face proportions visibly different
across five views, with stable centering/background and no clipping. The single
cylinder smoke showed a complete object with a visible top ellipse. This does
not replace the 45-image full contact-sheet review and does not authorize Fold
training.

## Smoke and contact-sheet review checklist

- Confirm scene metadata names `multiview_render_v2` and
  `clevr_neutral_fixed_v2`, while world/ground RGBA values exactly match v1.
- Confirm Blender 4.2.11, Cycles, exactly one visible V100 on GPU 3, 512
  samples, and `cycles_seed == render_seed`.
- For at least a cube and a sphere/cylinder smoke, confirm
  `look_at_target == objects[0].3d_coords`, target projection is centered, and
  the object is complete with mask ratio in `0.005–0.90`.
- Inspect the 45-image contact sheet in rows, not only globally: shape, color,
  material, object scale, and background must remain fixed across views
  `0/4/8/12/16`.
- Cube: visible-face proportions should change clearly without clipping.
- Cylinder: top ellipse visibility and side/top proportions should change
  clearly without extreme top-down or grazing views.
- Sphere: silhouette may remain stable; require plausible movement of
  highlight, shading, and shadow rather than a contour change.
- Confirm each cell has 20 distinct RGB SHA-256 values; masks align with every
  silhouette and object/background masks are complements.
- Reject black images, background drift, object displacement, scale drift,
  excessive framing change, extreme elevation, or lighting artifacts.
- Compare v1 and v2 sheets side-by-side and record whether v2's cube/cylinder
  viewpoint difference is materially easier to see. Keep Fold training gated
  if that judgment is unclear.
