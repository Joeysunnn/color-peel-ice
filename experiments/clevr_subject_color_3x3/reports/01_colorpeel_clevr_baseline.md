# 01 — ColorPeel CLEVR 3×3 baseline

- Status: **completed with explicit evaluation failures**
- Study: `clevr_subject_color_3x3`
- Method: `colorpeel_ice`
- Config: `../configs/baseline.yaml`
- Manifest: `../manifests/clevr_3x3_manifest.json`
- Run path: `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-130816__clevr_subject_color_3x3__baseline__c8c874d__42`
- Commit: `c8c874d00318ae7c1df2265c8627787d316a1ce3`, descending from baseline `021f5c74cee6c231a03b8b49bb96750cadfc4e06`

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
| `baseline` | none | official-parameter ColorPeel-on-CLEVR anchor | confirmed: 1500/1500 |
| `smoke2-first-two` | first two samples, 2 steps | startup plus seen-token observation | confirmed |
| `smoke9-full-grid` | all nine samples once, 9 steps | complete six-token exposure observation | confirmed |

No post-hoc variant may be added without documenting its rationale and control here before launch.

## Locked optimizer decision

Use literal official AdamW behavior. Ordinary vocabulary embedding drift is allowed, measured, and reported; it is not restored and is not a failure criterion. This is a comparability observation, not an unresolved blocker.

## Pre-run evidence boundary

Before creating a real smoke/full run, require:

1. clean Git commit and tracked config provenance;
2. 48-sample/nine-selection data audit and staged concepts hash;
3. pinned environment and exact SD 1.4 provenance;
4. launcher preflight in a separate consumed dry-run directory;
5. observation instrumentation capable of recording exposure counts, per-token nonzero-gradient steps, modifier deltas, and ordinary-vocabulary drift;
6. explicit stage records for later Grounded-SAM and Qwen3-VL execution.

Missing pre-run evidence keeps the run `pending`; a launcher dry-run is not a training smoke.

## Post-run evidence boundary

A process return code alone does not complete a smoke or full run. Append exact immutable paths for the run manifest, command, environment, stdout, token observations, checkpoints/embeddings, and reload evidence. Missing artifacts remain `pending`; they must not be reconstructed from memory.

For the two-step smoke, only exposed tokens require learning-signal evidence. For the nine-step smoke, each of the six tokens requires exposure count 3, at least one nonzero-gradient step, and nonzero final delta. Ordinary vocabulary drift is always descriptive, never pass/fail.

## Required evidence

| Evidence | Required source | Status |
|---|---|---|
| Clean Git provenance | run `manifest.json` | confirmed at `c8c874d` |
| Exact environment | `environment.txt` | confirmed and frozen |
| Data inventory and hashes | data audit plus locked manifest | confirmed: 48/48, selected 9 |
| Two-step seen-token observations | `checkpoints/training_metrics.jsonl`, `checkpoints/embedding_update_audit.json`, validator output | confirmed |
| Nine-step six-token observations | `checkpoints/training_metrics.jsonl`, `checkpoints/embedding_update_audit.json`, validator output | confirmed |
| Ordinary-vocabulary AdamW drift | both smoke observation outputs; record only | observed, not enforced |
| Finite diffusion/CAA losses | both smoke logs and full log | confirmed: 2 + 9 + 1500 rows |
| Reloadable checkpoint | checkpoint inventory and reload log | confirmed |
| 900 generation rows | generation manifest | confirmed: 900 valid images |
| Shape/color/joint accuracy | deterministic scorer output | observed: 94.44% / 93.89% / 93.89% |
| Official color metrics and segmentation failures | metric CSV and failure ledger | observed: 588 scored, 12 excluded/reported |
| Two axis contingency tables | scorer output | confirmed and reported below |
| Grounded-SAM masks/failures | independent Grounded-SAM stage manifest | 588 ok, 12 explicit failures |
| Qwen predictions/failures | independent Qwen3-VL stage manifest | 300 ok, 0 failures |

## Findings

- **confirmed**: both real training smokes and the full 1500-step run passed.
- **observed**: every modifier token received its exact expected exposure and
  nonzero-gradient count, and all six final embedding deltas were nonzero.
- **observed**: ordinary vocabulary rows drifted under literal official AdamW;
  this was recorded and not treated as a failure.
- **confirmed**: 900/900 generated images passed decode/RGB/size validation.
- **observed**: grid shape/color/joint accuracy was
  94.44%/93.89%/93.89% over 180 samples.
- **observed**: Grounded-SAM accepted 588/600 transfer masks; the 12 failures
  are split evenly between no detection and ratio rejection. The color scorer
  contains 588 valid rows plus 12 `mask_missing` rows.
- **observed**: overall 10%/50%/100% ΔE means were
  22.0011/31.3093/42.5375; corresponding ΔECh means were
  16.3569/22.6487/30.8198.
- **observed**: subject-only predictions were cube→gray 20/20,
  cylinder→gray 20/20, sphere→cyan 2/20 and other 18/20. Color-only rows
  (cube/sphere/cylinder/other/missing) were red 1/2/2/15/0, cyan
  10/2/1/7/0, and gray 3/4/2/11/0.
- **interpretation**: strong grid accuracy coexists with substantial
  single-axis output bias. No custom scalar leakage score or “entanglement
  solved” claim is introduced.

When evidence arrives, append findings with exact artifact paths. Do not rewrite a pending or superseded observation without preserving its history.

## Literature links

- [ColorPeel](../../../literature/notes/2024_butt_colorpeel.md)
- [ICE](../../../literature/notes/2025_cendra_ice.md)
