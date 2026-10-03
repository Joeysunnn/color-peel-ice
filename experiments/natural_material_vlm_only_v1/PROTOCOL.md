# VLM-only natural material baseline

Use the same natural mailbox photo as `natural_material_supermat_v1`, with its
subject eroded mask to select the visible target. The existing bare-metal spoon
is a separate positive control. A neutral gray masked crop is sent to the
locally cached Qwen3-VL-8B-Instruct model. Deterministic decoding returns one
label and a short visual cue. The complete prompt, raw response, crop, hashes,
model snapshot, and code commit are recorded in a new server run directory.

The closed labels and renderer lookup are fixed in `config.json` **before**
reading the predictions. The lookup's Roughness and Metallic numbers are
illustrative presets, not physical measurements or SuperMat outputs. If the
VLM returns `uncertain`, render nothing for that sample. The original photo's
color is excluded from all renderer presets; use the existing neutral-gray
Cycles sphere and three lights for visual comparison with the SuperMat run.

This baseline tests whether a semantic material description alone gives a
useful visual counterfactual. It cannot establish the original material's PBR
parameters or physical substrate. Keep the mailbox and spoon results separate.
Do not generate the 96-image grid or train `<M*>` from this protocol.
