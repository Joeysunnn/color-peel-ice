# Semantic constraint on SuperMat material extraction

Run Qwen3-VL on masked natural-image crops to describe visible surface family,
gloss, and opacity using closed categorical values. Run the same official
SuperMat checkpoint on the original object RGBA inputs. Do not feed VLM words
or renderer presets to SuperMat. Exclude Albedo from the canonical material.

The rules in `config.json` are fixed before inference. For each region, read
the median roughness and metallic from the SuperMat maps. Reject a region if
its map spread exceeds the broad quality gate or its medians conflict with
the categorical family/gloss gates. Select the accepted region with the
smallest sum of roughness and metallic IQRs. If none pass, return `rejected`
and no M*. The VLM never assigns, interpolates, clamps, or replaces R/M.

Opacity is recorded but cannot be validated: SuperMat provides no transmission
map in this protocol. `coated_or_painted` means the *visible finish*; it does
not prove that the hidden substrate is metal. The numeric gates are heuristic
consistency checks, not learned calibration or physical truth.

The mailbox has small painted-lid and subject eroded candidate masks. The
bare-metal spoon is a separate positive control with one bowl-interior mask.
Preserve VLM raw answers, SuperMat maps, candidate statistics, decisions,
source and model hashes, renderer metadata, and git commit in a new run root.
Use the existing three-light sphere renderer only for a small review sheet.
Do not generate the full grid or train `<M*>`.
