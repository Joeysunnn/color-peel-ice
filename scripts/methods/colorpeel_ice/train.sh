#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-CompVis/stable-diffusion-v1-4}"
CONCEPTS_LIST="${CONCEPTS_LIST:?Set CONCEPTS_LIST to the staged concepts.json outside the repository}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR below COLORPEEL_RUN_ROOT}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1500}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" CUDA_LAUNCH_BLOCKING=1 \
python src/train/train_colorpeel.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --output_dir="$OUT_DIR" \
  --concepts_list="$CONCEPTS_LIST" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=0 \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --checkpointing_steps="$CHECKPOINTING_STEPS" \
  --seed=42 \
  --mixed_precision=no \
  --cos_weight=0.2 \
  --scale_lr \
  --hflip \
  --modifier_token="<s1*>+<s2*>+<s3*>+<c1*>+<c2*>+<c3*>" \
  --initializer_token="cube+sphere+cylinder+red+cyan+gray"
