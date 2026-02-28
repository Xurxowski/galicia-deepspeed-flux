#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-black-forest-labs/FLUX.1-schnell}"
DATA_DIR="${DATA_DIR:-/tmp/galicia_dataset/train}"
DATASET_NAME="${DATASET_NAME:-}"
IMAGE_COLUMN="${IMAGE_COLUMN:-image}"
CAPTION_COLUMN="${CAPTION_COLUMN:-text}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/flux-galicia-lora}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-ethnographic photo of galician architecture and monuments}"
BASE_MODEL_LOWER="$(echo "${BASE_MODEL}" | tr '[:upper:]' '[:lower:]')"

RANK="${RANK:-16}"
RESOLUTION="${RESOLUTION:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1200}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-200}"
SEED="${SEED:-42}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"   # set fp16 for T4/MPS; bf16 for A10/L4/A100 etc.
USE_DEEPSPEED="${USE_DEEPSPEED:-auto}"      # auto|1|0
CACHE_LATENTS="${CACHE_LATENTS:-0}"         # 1 enables --cache_latents (less VRAM, more RAM/time)
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"  # 1 enables --gradient_checkpointing
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"  # keep 0 on laptops to reduce memory pressure
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"    # optional raw args passed to train script

DIFFUSERS_DIR="${DIFFUSERS_DIR:-/tmp/diffusers}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${DIFFUSERS_DIR}/examples/dreambooth/train_dreambooth_lora_flux.py}"
FORCE_DIFFUSERS_INSTALL="${FORCE_DIFFUSERS_INSTALL:-0}"

if [[ "${BASE_MODEL_LOWER}" == *"z-image"* ]]; then
  echo "BASE_MODEL=${BASE_MODEL} is from the Z-Image family." >&2
  echo "This training launcher is FLUX-specific (train_dreambooth_lora_flux.py)." >&2
  echo "Use FLUX here, or switch to a dedicated Z-Image fine-tuning recipe." >&2
  exit 1
fi

if [[ ! -d "${DIFFUSERS_DIR}" ]]; then
  git clone --depth 1 https://github.com/huggingface/diffusers "${DIFFUSERS_DIR}"
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

NEED_DIFFUSERS_INSTALL=0
if [[ "${FORCE_DIFFUSERS_INSTALL}" == "1" ]]; then
  NEED_DIFFUSERS_INSTALL=1
elif ! python - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("diffusers") else 1)
PY
then
  NEED_DIFFUSERS_INSTALL=1
fi

if [[ "${NEED_DIFFUSERS_INSTALL}" == "1" ]]; then
  python -m pip install -U pip
  python -m pip install --no-build-isolation -e "${DIFFUSERS_DIR}"
else
  echo "Using existing diffusers installation; skipping pip install."
fi

mkdir -p "${OUTPUT_DIR}"

DEVICE_KIND="$(
  python - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"

if [[ "${USE_DEEPSPEED}" == "auto" ]]; then
  if [[ "${DEVICE_KIND}" == "cuda" ]]; then
    USE_DEEPSPEED="1"
  else
    USE_DEEPSPEED="0"
  fi
fi

if [[ "${DEVICE_KIND}" == "mps" && "${MIXED_PRECISION}" == "bf16" ]]; then
  # PyTorch MPS does not support bf16 training.
  MIXED_PRECISION="fp16"
fi

if [[ "${DEVICE_KIND}" == "cpu" && "${MIXED_PRECISION}" != "no" ]]; then
  MIXED_PRECISION="no"
fi

TRAIN_ARGS=(
  --pretrained_model_name_or_path "${BASE_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --instance_prompt "${INSTANCE_PROMPT}"
  --resolution "${RESOLUTION}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler constant
  --lr_warmup_steps 0
  --rank "${RANK}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --seed "${SEED}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

if [[ "${CACHE_LATENTS}" == "1" ]]; then
  TRAIN_ARGS+=(--cache_latents)
fi

if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  TRAIN_ARGS+=(--gradient_checkpointing)
fi

if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # Allow advanced tuning flags without editing this launcher.
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_TRAIN_ARGS})
  TRAIN_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

if [[ -n "${DATASET_NAME}" ]]; then
  echo "Using dataset_name mode with captions:"
  echo "  DATASET_NAME=${DATASET_NAME}"
  echo "  IMAGE_COLUMN=${IMAGE_COLUMN}"
  echo "  CAPTION_COLUMN=${CAPTION_COLUMN}"
  TRAIN_ARGS+=(
    --dataset_name "${DATASET_NAME}"
    --image_column "${IMAGE_COLUMN}"
    --caption_column "${CAPTION_COLUMN}"
  )
else
  echo "Using instance_data_dir mode:"
  echo "  DATA_DIR=${DATA_DIR}"
  TRAIN_ARGS+=(--instance_data_dir "${DATA_DIR}")
fi

echo "Detected device: ${DEVICE_KIND}"
echo "MIXED_PRECISION=${MIXED_PRECISION}"
echo "USE_DEEPSPEED=${USE_DEEPSPEED}"
echo "CACHE_LATENTS=${CACHE_LATENTS}"
echo "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"

if [[ "${DEVICE_KIND}" == "mps" ]]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  accelerate launch \
    --config_file "$(
      if [[ "${MIXED_PRECISION}" == "fp16" ]]; then
        echo "${PROJECT_ROOT}/configs/accelerate_deepspeed_zero2_fp16.yaml"
      else
        echo "${PROJECT_ROOT}/configs/accelerate_deepspeed_zero2.yaml"
      fi
    )" \
    "${TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}"
else
  accelerate launch \
    --num_processes 1 \
    --mixed_precision "${MIXED_PRECISION}" \
    "${TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}"
fi

echo "Training finished. Artifacts at: ${OUTPUT_DIR}"
