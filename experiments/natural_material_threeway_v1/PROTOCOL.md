# Three source materials, Scheme 3

Use the previously accepted Scheme 3 mailbox and metal spoon material estimates
without editing their source runs. Add the uploaded wood spoon as an independent
third source. Qwen3-VL supplies categorical surface, gloss, and opacity priors;
SuperMat supplies roughness and metallic maps. VLM never sets numerical R/M.
The material concept is exactly the median `(roughness, metallic)` over a
documented material mask. Albedo color and wood grain are excluded.

Render a deterministic Cycles counterfactual grid with fixed R/M per source
while shape, base color, lighting, and viewpoint vary. Reserve an unseen shape
for transfer tests. Train three independent `<M*>` token-local K/V adapters on
the rendered training split, using the existing SD1.4 trainer. Evaluate each
adapter on held-out prompts and compositions with fixed seeds and base-model
controls. Keep extraction, rendering, training, and evaluation in a new run
directory; never modify earlier results.

The SuperMat-to-Cycles round-trip discrepancy is recorded as a limitation,
not used to adjust R/M. Wood appearance is evaluated as matte or glossy
nonmetal reflectance only; grain and its source color cannot be transferred
by this two-scalar material representation.
