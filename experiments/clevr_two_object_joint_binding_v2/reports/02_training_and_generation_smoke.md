# Joint-binding training and generation smoke

## Runtime evidence

- Code commit used for training: `affe4a554536ef9c73e57a83037b529f5bda35aa`.
- Server regression tests before training: 162 passed.
- 2-step smoke: passed; all reconstruction, within-bundle CAA, ICE
  Wasserstein, total-loss, gradient, save, and reload checks were finite.
- 18-step smoke: passed; subject/color exposure was 12 per token, material
  exposure was 18 per token, and all eight token rows had nonzero gradients.
- Full training run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/20260902-095000__clevr_two_object_joint_binding_v2__joint_binding_seed42__affe4a5__42`
- Full run completed 1500/1500 steps with zero non-finite metric rows.
- Mean reconstruction / CAA / weighted ICE attention / total loss:
  `0.419370 / 0.020196 / 0.000029745 / 0.423439`.
- Mean observed cross-object attention mass: `0.072126` (audit only).
- Accelerator checkpoint-1000, final Custom Diffusion K/V, all eight token
  files, embedding audit, and final pipeline reload succeeded.
- Literal official AdamW behavior remained enabled; ordinary vocabulary drift
  was recorded and not bounded.

The generation entrypoint initially rejected the new versioned training variant
before loading a model. Commit `1615d80` changes only the provenance allow-list,
adds its regression test, and does not change sampling. All 163 server tests
passed after that commit.

## Four-image generation gate

Generation run:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/20260902-101000__clevr_two_object_joint_binding_v2__joint_binding_generate_smoke_seed42__1615d80__42`

All four requested files were generated successfully at 512x512 using the
locked seed-42, 100-step, CFG-6.0 protocol. Compared with the immutable original
composition-failure baseline, every image contains two separable objects and no
3--5 object proliferation was observed.

The smoke nevertheless fails the semantic full-generation gate:

- `seen_pair_00_forward_seed_42`: requested left red-metal cube and right
  cyan-rubber sphere; the right sphere inherits red/metal-like appearance.
  SHA-256 `82ae735a29f80a5211db57d2652a61b2ae34d210e5308b1129661de9bf1218363`.
- `seen_pair_00_swapped_seed_42`: left cyan-rubber sphere and right red-metal
  cube are broadly correct; the sphere remains unusually reflective.
  SHA-256 `168074a18496829a450850b5445956d767fcc76c93619aed54009f5139f4c6b8d`.
- `unseen_pair_00_forward_seed_42`: requested left red-metal cube and right
  cyan-rubber cylinder; color/shape bundles are spatially swapped or malformed.
  SHA-256 `495e0d5a957fbc641f7132b9af36c1b0b7fecf4426626801bb912b87a913adf8e`.
- `unseen_pair_00_swapped_seed_42`: cyan-rubber cylinder is broadly correct,
  but the requested right red-metal cube is cylinder-like.
  SHA-256 `cd4139ed791849aa6b4fb3a26701bb09979561736f1341e566c6d56479d631200`.

Conclusion: joint prompt training plus two-mask guidance fixes object-count and
separation failure, but does not yet provide reliable left/right semantic bundle
binding. The 720-image campaign remains blocked pending human review and a
decision on the next binding intervention.
