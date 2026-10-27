# Two full-spoon counterfactual datasets

The user selected the original five colors: 180 images per subject, 360 total.
Each subject has its own source photograph, generated native geometry and camera.
Source names `metal_spoon` and `wood_spoon` identify different objects, not fixed
material labels. Captions use `spoon` and the actual rendered M/R controls.

The previous SuperMat spoon masks cover bowl crops only. This experiment uses
the tracked full-resolution original photographs, checked against fixed hashes,
and the official TRELLIS BiRefNet preprocessing model at a pinned revision.
Soft foreground alpha is resized without changing aspect ratio and padded to
512 square. Transparent RGB is zero. Original photographs and segmentation
previews are retained for inspection, including the complete handles.

`run_assets.sh NEW_EXTERNAL_ROOT` executes segmentation once, then two independent
seed42 1024-cascade reconstructions in separate processes. Reuse the verified
V100 FP16 flow conversion, xformers, explicit_gemm and converted same-model DINOv3
encoder. TRELLIS model revisions match the previous successful Mailbox run.
Save the complete native mesh and source-view native PBR/clay/BC/M/R reviews;
skip GLB UV baking because dataset rendering uses native vertices/faces directly.

After asset review, freeze each native asset and camera hash in a separate
protocol JSON. Render with `experiments.subject_trellis_dataset_v1.build
--protocol PATH --output NEW_EXTERNAL_DATASET`. Preserve the Mailbox five RGB
colors, M={0,1}, R={.15,.50,.85}, yaw offsets={-.25,0,.25}, studio/courtyard HDRIs,
512 resolution, constant background, full-frame camera fit and seed42 SSAO.
Each subject supplies 30 matched six-state groups and 210 single-factor edges.

All changes follow local edit/test/commit/push, then server Git pull and execution
from a clean dedicated checkout. Artifacts and logs stay outside Git. Never
overwrite failed or historical outputs. This task starts no training or monitor.

These are synthetic counterfactuals of generated geometry. Uniform Base Color
and scalar M/R replace all native appearance: original wood grain, printed logo
and mirror highlights are not preserved as texture. Single-image reconstruction
can alter shape, thickness, holes and hidden surfaces. Native/source-view reviews
and technical dataset checks support inspection; they do not prove photographic
identity fidelity or learned subject/material disentanglement.
