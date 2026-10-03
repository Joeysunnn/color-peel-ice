# Subject eroded mask extraction result

Run: `/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_supermat_v1/large_mask_subject_eroded_v1/`

The original natural mailbox image and full object RGBA input were held fixed.
The SuperMat checkpoint, seed 42, renderer profile, sphere, and three lights
were also held fixed. Only the region used to aggregate the SuperMat maps
changed from the small painted-lid mask to the subject eroded mask. The input
RGBA SHA256 is `1ba1e5010c6732b9cd241f6bdf7cdaee4404210b2e9e89e65d822ff713433670`;
all four new source SuperMat maps have the same hashes as the pilot maps.

| Measurement region | Pixels | Roughness median | Metallic median | Roughness IQR | Metallic IQR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small painted-lid region | 2,815 | 0.352941 | 0.039216 | 0.007843 | 0.003922 |
| Subject eroded mask | 40,644 | 0.356863 | 0.039216 | 0.039216 | 0.007843 |

The new extracted values were passed directly into Cycles Principled BSDF.
There was no roughness sweep or metallic override.

| Light | Pilot R-hat | New R-hat | Pilot M-hat | New M-hat |
| --- | ---: | ---: | ---: | ---: |
| Soft/front | 0.270588 | 0.274510 | 0.003922 | 0.003922 |
| Side/directional | 0.290196 | 0.290196 | 0.007843 | 0.007843 |
| Top/environment | 0.282353 | 0.282353 | 0.003922 | 0.003922 |

The larger aggregation region leaves the scalar estimate and round-trip bias
almost unchanged. Its wider roughness distribution also indicates that the
subject eroded mask mixes visible subparts rather than isolating one finish.
This rules out a too-small aggregation region as the sole explanation for the
observed bias. It does not test whether changing the SuperMat *input* alpha to
a homogeneous material region would change the prediction, and neither
SuperMat's estimate nor the round trip establishes physical ground truth.

This run is an instance of **SuperMat direct PBR extraction** with spatial
aggregation of roughness and metallic. Albedo color was excluded from the
canonical material. It is not a VLM, IID multi-hypothesis, or Material
Palette-style pipeline.

Review artifacts: `review/large_mask_subject_eroded_v1/` (local, ignored by Git).
The server run contains source and map hashes, SuperMat inference manifests,
renderer metadata, calibration images, and per-light round-trip reports.
The 96-image counterfactual grid and `<M*>` training remain unstarted.
