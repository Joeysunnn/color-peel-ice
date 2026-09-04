# Single-object checkpoint comparison result

## Evidence

The accepted single-object subject/color/material checkpoint and the joint-binding
checkpoint were compared under the locked complete-bundle diagnostic from
`03_single_object_diagnostic_protocol.md`. Both checkpoints generated the same
18 shape/color/material cells at seed 42, 100 inference steps, CFG 6.0, FP16 on
GPU 3, with the SafetyChecker disabled under the existing acknowledged policy.

The immutable run evidence is:

- joint checkpoint smoke:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/20260902-172000__clevr_two_object_joint_binding_v2__single_object_bundle_smoke_joint_seed42__7f9f302__42`;
- accepted single-object reference smoke:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/20260903-122500__clevr_two_object_joint_binding_v2__single_object_bundle_smoke_reference_seed42__7f9f302__42`;
- 36-image checkpoint comparison:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/20260903-122900__clevr_two_object_joint_binding_v2__single_object_checkpoint_comparison__7f9f302__42`.

## Human review verdict

The accepted single-object baseline was judged to perform well overall. Only a
very small number of samples showed mild material layering. In the paired
comparison, the joint-binding checkpoint added visibly more deformation,
strengthened the upper/lower material split on red spheres, and made gray rubber
appear more metal-like.

The joint-binding checkpoint therefore does not preserve the accepted
single-object token quality. Its improvement in two-object count and separation
remains useful negative-ablation evidence, but it is not an acceptable checkpoint
for further training or multi-object expansion.

## Decision and gates

- Freeze joint-binding v2 as a negative ablation. Do not retrain it or extend it.
- Use the accepted single-object `full_grid_seed42` checkpoint as the only
  semantic-token base for the next method design.
- Keep the 720-image two-object campaign and both 360-image checkpoint expansions
  blocked.
- Do not start Qwen, Grounded-SAM, automatic scoring, natural-image expansion, or
  any new training from this result.
- A separate mask-guided regional-inference method must be specified, reviewed,
  and pass its own minimal smoke and human gate before any larger campaign.

This result does not establish that ICE entanglement is solved. It establishes
that joint two-object fine-tuning damaged previously accepted single-object
semantics and must not be used as the next checkpoint base.
