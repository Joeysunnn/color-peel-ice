# Full-map material experiment status

Server run root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_fullmaps_v1/run_20261004_fullmaps_threeway_lighting_v2/`

SuperMat's reviewed visible regions from the three-source Scheme 3 run were
preserved as **Albedo RGB, Roughness, and Metallic maps**. Original maps and
masks, inpainted texture patches, crop coordinates, and hashes are in
`maps/`. VLM priors are provenance only. The paper's multiview UV refinement
is unavailable from these single natural images; box projection is the
documented approximation. Source color and visible wood grain are intentionally
part of the learned material concept.

The first full-map calibration run
`run_20261004_fullmaps_threeway_v1/` remains intact but is not training data.
Its uniform bright world made three lighting conditions nearly identical:
masked soft/side and soft/top mean differences were only 2-3/255. The final
lighting-v2 profile gives 25.9/255 and 30.2/255 on the metal sphere, while
retaining visible metal and wood appearances. Its three grids each contain
96 images. Pair verification passed: 96 shared conditions, identical object
masks, 72 train cases, and 24 held-out cone cases per material.

Original token-local K/V `<M*>` training was launched at server timestamp
`20261004-114311`, seed 42, 5000 steps, one independent run per material:

| Source | Training run directory |
| --- | --- |
| Mailbox | `20261004-114311__natural_material_fullmaps_v1__mailbox_fullmaps_token_local_kv_5000__241c32a__42` |
| Metal spoon | `20261004-114311__natural_material_fullmaps_v1__metal_spoon_fullmaps_token_local_kv_5000__241c32a__42` |
| Wood spoon | `20261004-114311__natural_material_fullmaps_v1__wood_spoon_fullmaps_token_local_kv_5000__241c32a__42` |

The `wood_spoon` launcher dry-run passed before launch. Training completion
and transfer results are **pending**. Per the user's preference, training is
not being monitored; continue evaluation when the user reports completion.
The fixed transfer protocol and scripts are ready in this experiment folder.
