# Pink Mailbox with SAM identity and BiRefNet soft alpha

2026-11-05. The requested preview completed one fresh seed42 mesh from current
Mailbox SAM plus BiRefNet soft alpha, recolored pink. Producer:
`87efe3a87d6cb20f4b181523ba45e82bc66e0cf2` in a new clean GitHub checkout.
All 12 alpha-association tests passed on the server without skips.

The initial `d5b9d15` attempt stopped at overly strict component association,
before mesh generation. Its alpha and error remain intact. The fixed 1% overlap
rule ignores the 43-pixel weak door contact for ownership, retaining the entire
44831-pixel Mailbox alpha component and 100% of its 42397-pixel SAM support. It
removes the separate number-plate component whole; no alpha normalization,
hard-SAM clipping or manual pixel editing is performed.

| Measurement | New pink + soft alpha |
| --- | ---: |
| Vertices | 1033002 |
| Faces | 2067646 |
| Review silhouette IoU against soft alpha | 0.8288188416 |
| Saved native silhouette IoU against SAM | 0.8190070199995758 |
| Camera refinement | Not converged, 422 evaluations |

The native silhouette measurement above is from the saved review mask, not the
new selector's separately rendered opaque geometry mask. Do not mix the two.
Source-view overlap passes 0.80, but inspection still shows a substantially
different backplate/tube relationship from the photograph. This is an example
of why silhouette overlap alone cannot establish correct 3D geometry. The user
has been shown the input, PBR and clay panels; quality is not marked accepted.

Server output:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/mailbox_soft_alpha_v1/20261105__sam_soft_pink__87efe3a__42/`.
Native mesh: `trellis/native_asset.pt`, SHA256
`133afbc08d3932a07fbf4681e9b4626a38a8df0cd1cbb9c571b0ec1136b65380`.
Input and alpha receipts, review images, camera trajectory and logs are retained.
No S11 expansion or training ran.

Local preview/evidence root:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/mailbox_soft_alpha_20261105/`.
`pink_soft_mailbox_preview.png` uses the same source-canvas crop
`[0,240,512,512]` for all three panels; complete images are in `evidence/trellis/review/`.
Archive SHA256:
`bc153580d9aaad85c6b4a1a5f3e0d172c9e6640d3d4ebc61a090b09470947643`.
