# Diagnose near-one TRELLIS roughness before comparing model quality

These independent diagnostic entry points do not change the original generation,
checkpoints, renderer, Material8 data or training defaults. No training is run.
Use the existing orange comparison's `trellis/` as `--source-run`; each output
must be a new external directory. Keep all stages and failed attempts.

- `weights`: CPU audit of all seven locally cached models. Verify recorded file
  hashes and HF cache blob identifiers; capture actual official loading's
  missing/unexpected keys, shapes and exact loaded tensors after dtype casting.
  Audit flow FP16 conversion and the six texture output-head rows. The cache
  blob check is local provenance, not an independent remote LFS query.
- `channels`: saved native six-channel statistics and official layout; replay
  raw attribute projection and compare to the old floating-point export. Two
  independent cloned-attribute sentinels verify BC/M/R/alpha channel responses.
  Report full-ROI direct errors, alpha-normalized errors and high-support errors.
  No original geometry or saved attributes are changed.
- `encoder`: current orange condition through the actual saved/reloaded official
  DINO extractor versus the pinned public timm reference at512/1024, using the
  exact same captured normalized pixel tensor. No new checkpoint downloads.
- `precision`: one fixed-input/seed inference with FP32 flows instead of the V100
  FP16 flow adaptation; unchanged weights/samplers/decoder. Then compare FP16
  and FP32 texture decoders on precisely the same latent and guide coordinates.
  This is a numeric diagnostic, not a new selected-quality generation protocol.

Run from the same clean Git commit with the existing TRELLIS environment:

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
PY=/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python
$PY -B -u -m experiments.trellis_roughness_audit_v1.weights --source-run "$SOURCE_RUN" --output "$AUDIT/weights"
CUDA_VISIBLE_DEVICES=2 $PY -B -u -m experiments.trellis_roughness_audit_v1.channels --source-run "$SOURCE_RUN" --roi "$SOURCE_RUN/../maps/common_roi.png" --output "$AUDIT/channels"
CUDA_VISIBLE_DEVICES=3 $PY -B -u -m experiments.trellis_roughness_audit_v1.encoder --source-run "$SOURCE_RUN" --output "$AUDIT/encoder" --device cuda:0
CUDA_VISIBLE_DEVICES=2 $PY -B -u -m experiments.trellis_roughness_audit_v1.precision --source-run "$SOURCE_RUN" --output "$AUDIT/precision"
```

Check GPU occupancy before selecting physical devices. Complete the weights
audit before interpreting precision results. A clean loader audit does not
establish physical material accuracy; neither does a single photograph supply
ground-truth roughness. Final reports must distinguish verified paths from
remaining ambiguities in the generated material estimate.
