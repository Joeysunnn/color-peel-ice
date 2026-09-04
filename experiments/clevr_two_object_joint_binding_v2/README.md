# CLEVR two-object joint binding v2

This is an independent scientific variant. It does not overwrite the accepted
single-object or original per-object-mask two-object runs.

The only method changes relative to the failed prompt-composition baseline are:

1. every training sample conditions on both complete left/right token bundles;
2. reconstruction is independently normalized inside both GT instance masks;
3. ColorPeel CAA is computed separately inside each object bundle;
4. ICE Stage Two mask-to-attention Wasserstein guidance (`1e-5`) localizes each
   bundle to its matching instance.

The same eight shared subject/color/material tokens, SD1.4, Custom Diffusion K/V,
ColorPeel CAA weight, AdamW behavior, resolution, learning rate and 1500-step
schedule are retained. No learned object-slot token is added.

## Current status

The paired single-object checkpoint review found that this joint-binding variant
adds deformation and material confusion relative to the accepted single-object
`full_grid_seed42` checkpoint. Joint-binding v2 is therefore frozen as a negative
ablation and is not approved for retraining or expansion. See
[`reports/04_single_object_checkpoint_comparison_result.md`](reports/04_single_object_checkpoint_comparison_result.md)
for the run evidence, human verdict, and blocked follow-up stages.
