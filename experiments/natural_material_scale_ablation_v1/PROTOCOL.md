# Metal-spoon full-map object-scale ablation

Goal: test whether the large foreground object in the full-map training grid
causes the observed shape/scene leakage. Only `grid_config.object_scale`
changes from 2.0 to 1.3. The original shape-specific captions, source
SuperMat A/R/M maps, box projection, shapes, lights, azimuths, camera height
offsets, Blender scene, render profile, held-out cone split, training setup,
seed, and fixed transfer evaluation remain unchanged.

Use the completed source at
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_fullmaps_v1/run_20261004_fullmaps_threeway_lighting_v2/`.
The source geometry and scene assets are
`/home/r12user5/Documents/Jiawei/papers/CLEVER/image_generation/data/`.
The source scene SHA-256 is
`e9a6a0a51f377064ae78e6568fdb1c5a6c97db4e674b29682e345d677607eb82`.
Use Blender 4.2.11 at
`/home/r12user5/Documents/Jiawei/tools/blender-4.2.11-linux-x64/blender`.

First render a sphere smoke image into a new output root and compare its
object-mask occupancy with the original scale-2.0 view. If this gives a
smaller valid foreground without clipping, render the complete 96-case grid
for metal spoon into a new experiment root. Stage 72 train images with the
existing `natural_material_fullmaps_v1/stage_training.py`; hold out all 24
cone cases. Train the original token-local K/V `<M*>` implementation for 5000
steps with the original shape-specific captions and matched parameters.
Evaluate the final checkpoint on the fixed cone/mailbox plain/photo prompts,
seeds 42–46, PNDM 100, CFG 3.5. Evaluate early steps if final transfer alone
cannot separate size effects from late training.

The old CLEVR pilot's mask occupancy was about 4.5–8.6%, while the original
full-map training objects occupied about 12–27%. Scale 1.3 is a predeclared
test setting, not a value tuned on transfer images. Its actual mask occupancy
must be measured after rendering. This experiment does not claim to isolate
the entire CLEVR scene difference.

Do not overwrite the full-map or generic-caption experiment. Save hashes,
renderer metadata, staged image/mask manifests, training config and transfer
outputs under the new run root.
