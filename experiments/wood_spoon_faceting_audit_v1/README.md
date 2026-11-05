# Wood spoon geometry / shading / color diagnostic

User request: identify the cause of pronounced bowl facets in the fresh SAM +
pink mesh, compare the historical uncolored spoon, and test whether recoloring
caused the difference. This experiment does not change the production pipeline.

Frozen assets:

- Historical original RGB + BiRefNet soft alpha:
  `/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons_v1/20261028-013740__assets__2943bf6__42/wood_spoon`.
- Current SAM + pink:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_sam_pink_v1/20261105__current_sam_pink__7726b73__42/object_001/trellis`.
- Original-color source receipt with exactly the current SAM mask:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_sam_pink_v1/20261105__current_sam_pink__7726b73__42/object_001/input/source_receipt.json`.

The two existing meshes share source-photo SHA, seed42, official code/model
revisions, encoder receipt, FP16 adaptation, samplers and renderer settings.
They differ in color, mask/alpha preprocessing and estimated camera. Historical
mask IoU cannot isolate recoloring or establish geometry quality.

Bounded protocol, from a reviewed clean GitHub checkout after GPU occupancy
checks, with both output-root variables pointing to the current run root:

1. One original-color mesh with the unchanged current SAM receipt using
   `experiments.subject_trellis_counterfactual_v1.run`, the existing encoder,
   pinned `experiments/subject_trellis_spoons_v1/model_revisions.json`, seed42
   and `--skip-glb`. No model download, seed search or camera continuation.
2. Render the old, current pink, and current original-color assets using
   `experiments.wood_spoon_faceting_audit_v1.render`. For each asset retain
   its saved source-fit camera, use the same projection/rasterization, and
   compare geometric face normals against area-weighted vertex normals.
   Simple fixed illumination excludes native textures, roughness and SSAO.
   Vertices/faces are never changed or written back. This is a diagnostic
   shader, not a replacement for the official PBR renderer.
3. Save full images, bowl detail views, source/mesh hashes and diagnostic
   measurements to a fresh external directory. Record camera and ROI explicitly.
   Compare the two SAM runs as a single-seed recoloring intervention, not a
   universal ranking of colors or proof of accepted 3D geometry.

Only diagnostic files are added. No mesh smoothing, remeshing, production
rendering patch, S11 expansion, initializer inference or training is performed.
