#!/usr/bin/env bash
# Run only from the clean server checkout pulled from the reviewed local commit.
set -euo pipefail
test "$#" -eq 1
root="$1"
test ! -e "$root"
test -z "$(git status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export HF_ENDPOINT=https://hf-mirror.com
python=/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python
encoder=/var/tmp/colorpeel-r12user5/assets/trellis2_v1/dino_timm_v2/model
"$python" -u -B -m experiments.subject_trellis_spoons_v1.prepare --output "$root/inputs"
for subject in metal_spoon wood_spoon; do
  "$python" -u -B -m experiments.subject_trellis_counterfactual_v1.run \
    --output "$root/$subject" --encoder "$encoder" \
    --source-receipt "$root/inputs/$subject/source_receipt.json" --skip-glb \
    --model-revisions experiments/subject_trellis_spoons_v1/model_revisions.json
done
