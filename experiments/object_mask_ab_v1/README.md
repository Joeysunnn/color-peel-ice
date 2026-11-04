# Object-mask schemes A and B

Compare automatic subject-mask intake before four-branch data preparation.
This test does not train Subject/Color/Material/Texture checkpoints and does not
establish that arbitrary photographs can yet complete the derived-data pipeline.

| Scheme | Discovery | Mask refinement | Status |
| --- | --- | --- | --- |
| A | Qwen3-VL whole-object categories, then all Grounding DINO boxes | SAM box prompts, followed by duplicate/containment suppression | Existing Mailbox result preserved; partial intake |
| B | ConceptExpress SD2.1 self-attention clustering, EOT foreground filtering, constrained region merging | Three foreground points per coarse mask, then SAM | Candidate test; results recorded separately |

Scheme A's existing implementation and usage remain in
[sam_multi_object_intake_v1](../sam_multi_object_intake_v1/README.md).
The preserved A commit is `4b537fac7d78946a0ffcf72c1c0d38e201b71ed7`.
Scheme B follows the inspected official ConceptExpress commit
[`db1dc61665866d208ba3a2e2152a83e7c6829e2f`](https://github.com/haoosz/ConceptExpress/tree/db1dc61665866d208ba3a2e2152a83e7c6829e2f).

## B test protocol

1. Use the same Mailbox photograph as A:
   `experiments/natural_material_supermat_v1/assets/source_mailbox.jpg`, SHA256
   `ba9c6624d9998cdfcd16308b0cf8fd8f3778909d8655efe2ba9cebba370483d0`.
2. Run one fixed-seed (`20`) extraction using cached SD2.1 base, empty prompt,
   timestep zero, and a 512x512 input without horizontal flipping. Preserve
   64x64 coarse masks and explicitly map them to original-image coordinates.
3. Sample three positive points from each mapped coarse mask. Use the same
   cached SAM ViT-base revision as A
   (`70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`); select the candidate with
   maximum SAM-predicted IoU. Preserve every coarse concept and its refinement,
   including overlaps, failures, sampled points and candidate scores.
4. Inspect full-image overlays and per-concept masks for whole-object coverage,
   background leakage, fragmentation and merged instances. Only after prediction
   may the historical Mailbox reference mask be read for descriptive IoU.
5. Stop after this bounded comparison. No seed/threshold search, GT-assisted
   point selection, automatic acceptance, or four-branch training is included.

The paper describes three random points but does not publish its complete SAM
reference-mask generation script. Seed 20, positive-point labels, reuse of SAM
ViT-base, and maximum predicted-IoU selection are explicit choices for this
candidate test; they are not claimed as recovered official SAM settings.
Any implementation or preprocessing deviation belongs in the run receipt.

The standalone extractor reuses FINCH and its helper functions from the pinned
external ConceptExpress checkout. It loads only those definitions to avoid
unrelated training dependencies; the external files remain unchanged and their
hashes are recorded. It preserves the official attention controller's last-half
head selection, including on this batch-one path. Explicit adaptations are
eval mode, no random flip, a standalone seeded VAE sample, and CPU aggregation.

## Run B

After local tests, review, commit/push and a fresh clean server checkout:

```bash
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT="$COLORPEEL_RUN_ROOT"
export PYTHONDONTWRITEBYTECODE=1
/home/r12user5/miniforge3/envs/ice/bin/python -B \
  experiments/object_mask_ab_v1/run.py \
  --image experiments/natural_material_supermat_v1/assets/source_mailbox.jpg \
  --upstream-checkout /home/r12user5/Documents/Jiawei/colorpeel-checkouts/conceptexpress-db1dc616 \
  --output "$COLORPEEL_RUN_ROOT/object_mask_ab_v1/NEW_RUN" --gpu 0 --seed 20
```

`status.json` and `objects.json` preserve execution state and per-concept SAM
scores. `attention_diagnostics.npz` stores EOT and pre-cluster labels;
`coarse/` stores latent/native masks, overlays and the sampled points;
`objects/` stores corresponding native SAM masks, overlays and isolated crops.
There is no category naming or initializer/training step in this bounded test.

Use reviewed, tested and pushed source in a fresh clean server checkout, with
the full commit recorded. Keep new run outputs outside Git under
`/home/r12user5/Documents/Jiawei/colorpeel-runs/object_mask_ab_v1/` and set both
`COLORPEEL_RUN_ROOT` and `COLORPEEL_OUTPUT_ROOT` to
`/home/r12user5/Documents/Jiawei/colorpeel-runs`. Check GPU processes and owners
before running; preserve prior A outputs and any failed B attempt.

## Interpretation

Automatic concept regions are not guaranteed to match all physical instances.
One Mailbox scene can reveal practical errors but cannot establish exhaustive
multi-object discovery. SAM-predicted IoU is a candidate-selection score, not
measured segmentation accuracy. The old Mailbox reference is only a limited
post-prediction comparison; it supplies no GT for the other discovered regions.

See [RESULTS.md](RESULTS.md) for outcomes and evidence locations.
