# Controlled Material initialization

The default single-image Qwen entry now emits `colorpeel_vlm_initializers/v2`.
Subject, Color and Texture retain their existing semantic selection rules.
Material emits only two separately evidenced classifications:

| Axis | Labels | Per-axis fields |
| --- | --- | --- |
| metallicity | metallic, dielectric, mixed, uncertain | label, confidence(high/medium/low), evidence |
| roughness | low, medium, high, uncertain | label, confidence(high/medium/low), evidence |

Classify visible reflection, not substrate or macroscopic texture. The original
response is immutable. The selected word, token ID, candidate audit, policy version,
selection reason and fallback flag are saved separately. Confidence is qualitative,
not a calibrated probability; the classifications do not measure PBR parameters.

`material_reflection_pool/v1` applies these ordered rules:

1. Mixed metallicity → `material` fallback (one scalar material cannot represent multiple regions).
2. High-confidence metallic → `metallic`.
3. Low roughness with medium/high confidence → `glossy`.
4. High roughness with medium/high confidence → `matte`.
5. Otherwise → `material` fallback, including medium roughness and insufficient evidence.

Only the selected single ordinary SD2.1 tokenizer token is copied. No free Material
word, first-BPE truncation, embedding averaging or implicit synonym substitution.
`painted`, color names, substrate names and processing/texture words cannot enter
through the controlled Material schema. A generic fallback is recorded explicitly;
it is not evidence that physical material identification succeeded.

VLM-v1 and explicit initializer-ablation manifests remain loadable for old runs.
Their original evidence and selected weights are unchanged. Future identification
uses v2 by default. Initializer copy, paired adapter/embedding checkpoint and
four-branch inference provenance checks continue to use the shared manifest loader.

## Material target consistency

The Mailbox dataset resolver verifies the original photograph against the SuperMat
input RGB, then compares the8 actual render records, metadata and staging values
against the same source/protocol R/M. All checked provenance is hash-locked.
`dataset/material_target_consistency.json` records the result before training.

Semantic comparisons are deliberately broad, versioned diagnostic heuristics.
For either axis, only high-confidence contradictions block preparation:
high-confidence metallic versus M<=0.2 or dielectric versus M>=0.8; low roughness
versus R>=0.65 or high roughness versus R<=0.25 flag a clear conflict and block
automatic preparation. These thresholds are not a physical calibration. Weaker
evidence/uncertainty and medium roughness do not establish numerical agreement.
No result rewrites M/R, re-renders data, changes the initializer, or upgrades a VLM
confidence. Mixed surfaces are flagged as a limitation of scalar M/R.

## Usage

1. In existing `ice-vlm`, run `scripts/methods/colorpeel_ice/initialize_from_image.py`
   with the same `--image`, pinned `--model-dir`, `--sd-model-dir`, fresh `--output`,
   and an idle `--device`. It now produces the controlled v2 manifest.
2. In `colorpeel017`, prepare using the existing four-branch entry:
   `python -B -m experiments.sd21_qwen_initializers_v1.run --stage prepare --initializer-manifest VLM_OUTPUT/initializers.json --output FRESH_EXPERIMENT`.
3. Optional bounded integration check:
   `python -B -m experiments.sd21_controlled_material_v2.verify --experiment FRESH_EXPERIMENT`.
   This records all four startup configs, trains Material for2 updates on GPU2,
   verifies its paired export and generates one chair/seed30 transfer image.
   It does not assess transfer quality or start a full training round.
4. The existing `--stage train` entry remains available for a separately requested
   full four-branch run. The bounded check stores outputs in `verification/` and
   leaves the prepared experiment ready for that entry.

The derived-data resolver remains explicitly limited to the audited Mailbox source;
a new photograph needs its own processing and reviewed derived datasets.
