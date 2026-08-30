# Material held-out Fold training

## Outcome

The locked Fold campaign completed all nine runs at commit `dcb465c`:
Fold A/B/C x training seeds 42/43/44. Execution was serial on GPU 3 and would
have stopped on the first launcher or audit failure. The campaign ended with
`CAMPAIGN_SUCCESS` at 2026-08-30T14:16:14+08:00.

Campaign root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260830-125800__material_fold_training_campaign__dcb465c`

## Verification

- 9/9 launcher manifests: `succeeded`, return code 0.
- 13,500/13,500 total optimization steps.
- Every run has exactly 1500 sequential finite metric rows.
- Every run has a complete `checkpoint-1000`.
- Every run saved nonempty Custom Diffusion weights and all eight token files.
- Successful launcher exit occurred after loading the saved K/V and eight
  textual-inversion artifacts into the final pipeline.
- Every token had nonzero gradients on every exposure and a positive finite
  initial-to-final embedding delta.
- Across runs, per-token exposure counts ranged from 476 to 764 and embedding
  deltas ranged from 0.0155083546 to 0.0271381456.
- Observed total loss values were finite and ranged from 0.0022332191 to
  3.3468232155 across the campaign.
- Official AdamW changed all 49,408 ordinary vocabulary rows in each run; this
  remained record-only and was not restored or treated as failure.

No metric-based checkpoint selection was performed. Each run's final artifacts
are the evaluation checkpoint. No held-out image or disentanglement result is
claimed by training completion alone.

## Released next stage

The nine immutable held-out generation configs may now run. Each checkpoint
generates 18 prompts x seeds 42-61 = 360 images: 240 seen and 120 held-out.
The complete campaign is 3240 images. It must stop at the merged human-review
gate before Qwen or scoring.
