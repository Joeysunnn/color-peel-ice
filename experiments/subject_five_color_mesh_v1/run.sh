#!/usr/bin/env bash
set -euo pipefail

source_receipt=${1:?original source receipt required}
output=${2:?fresh external output required}
gpu=${3:-0}
ice_python=/home/r12user5/miniforge3/envs/ice/bin/python
trellis_python=/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python
test -z "$(git status --porcelain)"
output=$("$ice_python" -B - "$source_receipt" "$output" <<'PY'
from pathlib import Path
import sys
from experiments.subject_trellis_counterfactual_v1.run import ASSETS, check_source_receipt
from scripts.launch.colorpeel_run import resolve_run_dir
from src.methods.colorpeel_ice.single_image_subject import ROOT
source = Path(sys.argv[1]).resolve()
inputs = check_source_receipt(source)
output = resolve_run_dir(Path(sys.argv[2]))
protected = (ROOT, ASSETS, source.parent, *(Path(path).parent for path in inputs['sha256']))
if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
    raise ValueError('require fresh output separate from code, assets and sources')
print(output)
PY
)
mkdir -p -- "$output"
git rev-parse HEAD > "$output/code_commit.txt"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for spec in pink:0 green:120 cyan:180 blue:240 purple:300; do
    color=${spec%:*}
    hue=${spec#*:}
    "$ice_python" -B -c 'import sys; from experiments.sd21_qwen_initializers_v1.run import require_idle; require_idle([int(sys.argv[1])])' "$gpu"
    "$ice_python" -B -m experiments.subject_pink_input_repro_v1.prepare \
        --source-receipt "$source_receipt" --output "$output/$color/inputs" --hue-degrees "$hue"
    "$trellis_python" -u -B -m experiments.subject_trellis_counterfactual_v1.run \
        --output "$output/$color/trellis" --source-receipt "$output/$color/inputs/source_receipt.json" \
        --encoder /var/tmp/colorpeel-r12user5/assets/trellis2_v1/dino_timm_v2/model \
        --model-revisions experiments/subject_trellis_spoons_v1/model_revisions.json \
        --skip-glb --seed 42 > "$output/$color/trellis.log" 2>&1
    printf 'COMPLETED %s\n' "$color"
done
