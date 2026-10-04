# Natural material full-map pilot

This independent run starts from the same three natural images and SuperMat
outputs as `natural_material_threeway_v1`. It preserves SuperMat's complete
visible-region Albedo RGB, Roughness, and Metallic maps. The material concept
therefore includes source color and any visible pattern in Albedo. It does not
include a normal map: single-image SuperMat does not predict one.

For each source, select the previously reviewed material mask, crop the
smallest bounding rectangle, and inpaint only pixels outside that mask so a
rectangular texture can be sampled. Save original maps, mask, texture patches,
hashes, crop coordinates, and fill fraction. No VLM category adjusts numeric
maps. Existing Scheme 3 semantic labels are retained as provenance only.

Map the three texture patches to Blender Principled BSDF Base Color,
Roughness, and Metallic using the same object-space box projection. This is an
explicit single-view-to-3D approximation. SuperMat's paper uses multiview
images and UV refinement for 3D texture reconstruction; this run does not
claim to reproduce that component.

The inherited dark-world renderer made the high-metallic spoon nearly black.
A white studio world was therefore used. Uniform world strength 2.0 made the
three lighting cases nearly identical (mean masked pixel difference only
2-3/255); it is retained as a calibration run, not training data. The final
profile uses world strengths 1.0, 0.5, and 2.0, with the three existing lamp
arrangements. In a metal-sphere smoke check, the resulting soft/side and
soft/top masked differences were 25.9/255 and 30.2/255. Texture repeat is 1.0:
repeating the single-view patch four times introduced a visible grid on the
metal sphere.

Keep source Albedo fixed within each material. Vary shape, lighting, and
viewpoint. Do not apply the previous red/blue/green/yellow base-color sweep,
because changing Base Color would now change the material itself. Reserve
cone as unseen shape. Compare separately with the previous R/M-only run;
the changed data grid means the comparison is observational, not a controlled
single-variable ablation.

Training and transfer, if run, use new directories and the same original
token-local K/V implementation. Material prompts omit explicit color:
`a photo of a {shape} made of <M*>`. Identity composition uses the fixed
balanced-aligned `<S*>` checkpoint. Report visible source color/pattern,
reflectance, object identity, and leakage separately. Do not overwrite prior
maps, datasets, checkpoints, or evaluations.

Official method references:
[SuperMat code](https://github.com/hyj542682306/SuperMat/blob/main/inference_supermat.py)
and [paper](https://arxiv.org/html/2411.17515v4).
