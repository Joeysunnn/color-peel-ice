#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-colorpeel017}"
CONDA_BIN="${CONDA_BIN:-conda}"

"$CONDA_BIN" create --name "$ENV_NAME" --yes python=3.10 pip
"$CONDA_BIN" run --name "$ENV_NAME" python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117
"$CONDA_BIN" run --name "$ENV_NAME" python -m pip install \
  --requirement environment/requirements-colorpeel017.txt
"$CONDA_BIN" run --name "$ENV_NAME" python -m pip install \
  "git+https://github.com/huggingface/diffusers.git@v0.17.0"

"$CONDA_BIN" run --name "$ENV_NAME" python -m pip freeze
