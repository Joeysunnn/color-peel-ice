# Semantic constraint result

Run: `/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_semantic_constraint_v1/run_20261003_mailbox_spoon_v1/`

The semantic and numerical streams were run independently at project commit
`52c3a3f`. Qwen3-VL-8B-Instruct snapshot
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` used the masked natural
images, seed 42, FP16, and deterministic decoding. The official SuperMat CLI
used the original whole-object RGBA inputs and seed 42. Its source Albedo,
Roughness, Metallic, and ORM map hashes match the prior mailbox and spoon runs.
The VLM never saw SuperMat numerical results and SuperMat never saw VLM text.

| Sample | VLM family / gloss / opacity | Selected region | SuperMat R* | SuperMat M* |
| --- | --- | --- | ---: | ---: |
| Mailbox | coated_or_painted / semi_gloss / opaque | small_painted_lid | 0.352941 | 0.039216 |
| Spoon control | bare_metal / glossy / opaque | spoon_bowl_interior | 0.247059 | 0.847059 |

Both mailbox candidate regions passed the predeclared family, gloss, and
spread checks. The small painted-lid region won because its combined R/M IQR
is lower: R IQR 0.007843, M IQR 0.003922. The subject eroded mask yielded
R 0.356863, M 0.039216, R IQR 0.039216, M IQR 0.007843. The spoon candidate
also passed all applicable checks. Opacity is recorded only: neither the
SuperMat maps nor M*=(R,M) provide a transmission estimate.
As a gate sanity check, changing only the mailbox semantic family to
`bare_metal` rejects its low-metallic candidate; changing only the spoon
family to `coated_or_painted` rejects its high-metallic candidate.

The selected SuperMat medians were passed directly into the same three-light,
neutral-gray Cycles renderer. The six renders were sent through SuperMat again:

| Sample | Light | R-hat | M-hat |
| --- | --- | ---: | ---: |
| Mailbox | soft/front | 0.271 | 0.004 |
| Mailbox | side/directional | 0.290 | 0.008 |
| Mailbox | top/environment | 0.282 | 0.004 |
| Spoon | soft/front | 0.137 | 0.800 |
| Spoon | side/directional | 0.145 | 0.169 |
| Spoon | top/environment | 0.122 | 0.220 |

The semantic prior is compatible with the selected SuperMat estimates, but
it did **not** change the mailbox M* relative to direct extraction. The
mailbox roughness round-trip bias remains. The spoon's metallic round trip
changes greatly with lighting, even though the input to the renderer uses one
fixed metallic value. A passed semantic gate therefore does not establish
physical PBR accuracy or renderer-space consistency. The positive control
demonstrates a distinct high-metallic source estimate, not a universal
calibration. The VLM's painted/coated category describes the visible finish;
it cannot prove the substrate beneath the coating.

The server run contains raw VLM replies, input and model hashes, all SuperMat
maps, candidate masks and checks, canonical JSONs, renderer metadata, and
round-trip manifests. A local review copy is ignored by Git at
`review/run_20261003_mailbox_spoon_v1/`. The 96-image grid and `<M*>`
training remain unstarted.
