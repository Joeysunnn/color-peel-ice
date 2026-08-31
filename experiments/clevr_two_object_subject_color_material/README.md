# Controlled two-object CLEVR stage

This versioned stage extends the accepted single-object subject/color/material
protocol to two objects. It does not alter historical renderer profiles or
checkpoints.

The scientific change is deliberately narrow: every optimization sample uses
the full two-object RGB image plus one GT instance mask and one complete
subject/color/material token bundle. Both objects share the same eight learned
semantic tokens and the same Custom Diffusion model. CAA remains unchanged and
therefore only aligns the three tokens present in the current object bundle.

The first gate is renderer-only: one real Blender smoke followed by 360 scenes,
realization, and human review. No training is authorized before that gate.
