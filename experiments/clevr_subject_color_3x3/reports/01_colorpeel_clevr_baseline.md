# 01 — ColorPeel CLEVR 3×3 baseline

- Status: **pending**
- Study: `clevr_subject_color_3x3`
- Method: `colorpeel_ice`
- Config: `../configs/baseline.yaml`
- Manifest: `../manifests/clevr_3x3_manifest.json`
- Run path: pending
- Commit: pending; must descend from baseline `021f5c74cee6c231a03b8b49bb96750cadfc4e06`

## Hypothesis

The official ColorPeel mechanism can be transferred to the locked metal CLEVR 3×3 grid with three shared subject tokens and three shared color tokens, producing auditable evidence about subject/color leakage without adding ICE method components.

This hypothesis is not a claim that entanglement is solved.

## Fixed protocol

- Nine images: cube/sphere/cylinder × red/cyan/gray × metal.
- Six tokens: `<s1*>`, `<s2*>`, `<s3*>`, `<c1*>`, `<c2*>`, `<c3*>`.
- SD 1.4, Diffusers v0.17.0, Custom Diffusion K/V tuning, modifier embeddings, and CAA weight 0.2.
- Seed 42 for training; seeds 42–61 for generation/evaluation.
- 512 px, batch 1, 1500 steps, LR `1e-5`, constant scheduler, zero warmup.
- GT masks excluded from training.
- No ICE Stage Two, triplet loss, attention-mask loss, material token, ICE baseline, or alternative checkpoint.

## Included variants

| Variant | Changed settings | Intended control | Status |
|---|---|---|---|
| `baseline` | none | official-parameter ColorPeel-on-CLEVR anchor | pending |

No post-hoc variant may be added without documenting its rationale and control here before launch.

## Pre-run blockers

1. The locked manifest must pass the 48-sample/nine-selection audit.
2. SD 1.4 availability and the pinned environment must be verified.
3. AdamW weight decay may change non-modifier embedding values even after their gradients are zeroed.
4. A value-level optimizer-step test and human review must resolve item 3 before smoke training.

## Required evidence

| Evidence | Required source | Status |
|---|---|---|
| Clean Git provenance | run `manifest.json` | pending |
| Exact environment | `environment.txt` | pending |
| Data inventory and hashes | data audit plus locked manifest | pending |
| Six modifier updates and frozen non-modifier values | optimizer-step test and smoke log | pending |
| Finite diffusion/CAA losses | smoke/full log | pending |
| Reloadable checkpoint | checkpoint inventory and reload log | pending |
| 900 generation rows | generation manifest | pending |
| Shape/color/joint accuracy | deterministic scorer output | pending |
| Official color metrics and segmentation failures | metric CSV and failure ledger | pending |
| Two axis contingency tables | scorer output | pending |

## Findings

- **pending**: no smoke test or full training run is recorded.
- **pending**: no generated-image or metric evidence is recorded.
- **pending**: no conclusion about entanglement reduction is authorized.

When evidence arrives, append findings with exact artifact paths. Do not rewrite a pending or superseded observation without preserving its history.

## Literature links

- [ColorPeel](../../../literature/notes/2024_butt_colorpeel.md)
- [ICE](../../../literature/notes/2025_cendra_ice.md)
