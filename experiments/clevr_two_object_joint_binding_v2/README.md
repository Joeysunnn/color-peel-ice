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
