# Balanced Subject token-local K/V step ablation

This study repeats the caption-aligned balanced mailbox Subject baseline with
the existing full-rank `token_local_kv` method. The reviewed five original
metal images and five matte counterfactuals, captions, optimizer settings,
seed, and learning rates are unchanged.

One continuous 2,000-step trajectory saves exact training-state snapshots at
steps 300, 500, 700, 1,000, and 2,000. The locked training assets and method
settings are recorded in `protocols/training_v1.json`; `configs/train.yaml` is
the launcher input.

This is not a LoRA experiment. The only changes from
`mailbox_subject_balanced_caption_aligned_5000` are the shorter 2,000-step
training dose and the explicit snapshot plan.

After training, `evaluate.py` derives a complete inference checkpoint from one
of the five paired adapter/text-encoder snapshots. Its `comparison` task
replays the original 36-image full-rank Subject/Material diagnostic exactly:
red, blue, and city-street groups; Subject-only, Material-only, Subject plus
Material, and literal-metal conditions; and seeds 42--44. Material remains the
selected ground-reflection `<M*>` token-local K/V checkpoint at step 5,000.

The independent `transfer` task replays all 28 historical Subject-transfer
prompts at seeds 42--46, for 140 images per Subject snapshot. The locked prompt
and sampling inputs are in `protocols/inference_v1.json`; each output
provenance file records the resolved training run and snapshot hashes.
