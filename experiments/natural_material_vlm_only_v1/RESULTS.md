# VLM-only baseline result

Run: `/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_vlm_only_v1/run_20261003_mailbox_spoon_v1/`

The predeclared labels and renderer presets are in `config.json`. Qwen3-VL-8B-Instruct
snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` ran in the existing
`ice-vlm` environment on masked natural-image crops with seed 42, FP16,
`do_sample=False`, and 128 maximum output tokens. The inference project commit
was `117d1d7`. The run manifest and per-sample files preserve the complete
prompt, model response, input hashes, crop, and renderer metadata. The
environment used PyTorch `2.5.1+cu124` and Transformers `5.14.1`.

| Sample | VLM label | Fixed renderer R | Fixed renderer M |
| --- | --- | ---: | ---: |
| Natural mailbox | `painted_or_coated_metal` | 0.45 | 0.0 |
| Bare-metal spoon control | `bare_polished_metal` | 0.18 | 1.0 |

The same neutral-gray sphere and three-light Cycles profile used by the
SuperMat diagnostic rendered both presets. The spoon preset shows strong
specular metal highlights; the mailbox preset looks like a moderately glossy
coating. These are visual sanity checks of the lookup and renderer, not a
measurement of the natural materials. Source color was not copied to the
renderer. The mailbox VLM evidence text nevertheless mentioned its orange
hue despite the prompt asking it not to; only the selected label entered the
renderer lookup.

The label `painted_or_coated_metal` is a semantic guess about an underlying
substrate that is hidden by the coating. The image alone cannot prove it.
The preset numbers were chosen before inference and must never be interpreted
as VLM-predicted or physically recovered PBR values. In particular, this
baseline cannot replace the SuperMat direct extraction or establish a
round-trip PBR consistency result.

Review sheet and copied inference records: `review/run_20261003_mailbox_spoon_v1/`
(ignored by Git). No counterfactual grid or `<M*>` training was run.
