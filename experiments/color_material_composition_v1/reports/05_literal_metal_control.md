# Literal metal versus learned M control

The 9 matched comparisons in
`$COLORPEEL_RUN_ROOT/color_material_composition_v1/20260925-120928__literal_metal_control__e5b568a__42/output`
completed with the same objects, seeds 42–44, 100 sampling steps, CFG 3.5,
and literal `orange color`. The left prompt uses `metal material`; the right
uses `<M*> material`. The visual contact sheet is saved in the run parent as
`literal_metal_contact_sheet.jpg`.

Visual review:

- Cube seed 42/43: `<M*>` produces a more coherent cube; literal metal causes
  extra or separated geometry. Seed 44 fails the cube shape in both prompts.
- Sphere seed 42: both are orange; `<M*>` has stronger metallic lighting.
  Seed 43: both are mostly silver/brown with a narrow orange band. Seed 44:
  both are decorative open sculptures rather than simple spheres.
- Mug seed 42: both are orange but have weak metal appearance. Seed 43: both
  are silver metal with little orange on the mug itself. Seed 44: both are
  brown/copper and do not clearly preserve the intended orange color.

The learned material token is not the sole cause of failed orange-metal
composition. Literal metal also fails color or object identity in several
matched cases. The paired-data joint C/M ablation remains a hypothesis; any
CAA claim must come from the weight-0 versus weight-0.2 comparison under the
same joint architecture and data.
