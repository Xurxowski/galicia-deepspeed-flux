#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -d "${PROJECT_ROOT}/.venv/bin" ]]; then
  export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"
fi

HF_USERNAME="${HF_USERNAME:-Xurxowsky}"
SPACE_ID="${SPACE_ID:-${HF_USERNAME}/galicia-horreos-cruceiros}"
BASE_MODEL="${BASE_MODEL:-black-forest-labs/FLUX.1-schnell}"

DATA_DIR_HORREO="${DATA_DIR_HORREO:-/private/tmp/galicia_dataset_horreo_v2/train}"
DATA_DIR_CRUCEIRO="${DATA_DIR_CRUCEIRO:-/private/tmp/galicia_dataset_cruceiro_v2/train}"

OUTPUT_DIR_HORREO="${OUTPUT_DIR_HORREO:-/private/tmp/flux-galicia-lora-horreo}"
OUTPUT_DIR_CRUCEIRO="${OUTPUT_DIR_CRUCEIRO:-/private/tmp/flux-galicia-lora-cruceiro}"

MODEL_REPO_HORREO="${MODEL_REPO_HORREO:-${HF_USERNAME}/flux-schnell-galicia-lora-horreo}"
MODEL_REPO_CRUCEIRO="${MODEL_REPO_CRUCEIRO:-${HF_USERNAME}/flux-schnell-galicia-lora-cruceiro}"

MAX_TRAIN_STEPS_HORREO="${MAX_TRAIN_STEPS_HORREO:-200}"
MAX_TRAIN_STEPS_CRUCEIRO="${MAX_TRAIN_STEPS_CRUCEIRO:-300}"
RESOLUTION="${RESOLUTION:-512}"
RANK="${RANK:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(
    ./.venv/bin/python - <<'PY'
from huggingface_hub import get_token
print(get_token() or "")
PY
  )"
fi

if [[ -z "${HF_TOKEN}" ]]; then
  echo "Missing HF token. Run: hf auth login" >&2
  exit 1
fi

export HF_TOKEN
export HF_HOME="${HF_HOME:-/private/tmp/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

run_train_upload() {
  local label="$1"
  local data_dir="$2"
  local output_dir="$3"
  local model_repo="$4"
  local max_steps="$5"
  local instance_prompt="$6"

  echo "==> [${label}] training start"
  export BASE_MODEL
  export DATA_DIR="${data_dir}"
  export OUTPUT_DIR="${output_dir}"
  export INSTANCE_PROMPT="${instance_prompt}"
  export MAX_TRAIN_STEPS="${max_steps}"
  export RESOLUTION
  export RANK
  export GRAD_ACC
  export LEARNING_RATE
  export MIXED_PRECISION
  export USE_DEEPSPEED
  export CHECKPOINTING_STEPS=50

  bash scripts/train_flux_lora_deepspeed.sh

  echo "==> [${label}] uploading ${model_repo}"
  ./.venv/bin/python scripts/upload_to_hub.py \
    --repo_id "${model_repo}" \
    --repo_type model \
    --folder_path "${output_dir}"
}

run_train_upload \
  "horreo" \
  "${DATA_DIR_HORREO}" \
  "${OUTPUT_DIR_HORREO}" \
  "${MODEL_REPO_HORREO}" \
  "${MAX_TRAIN_STEPS_HORREO}" \
  "ethnographic documentary photo of a traditional galician horreo, raised granary on stone pillars, elongated slatted chamber, rural Galicia"

run_train_upload \
  "cruceiro" \
  "${DATA_DIR_CRUCEIRO}" \
  "${OUTPUT_DIR_CRUCEIRO}" \
  "${MODEL_REPO_CRUCEIRO}" \
  "${MAX_TRAIN_STEPS_CRUCEIRO}" \
  "ethnographic documentary photo of a galician cruceiro, carved granite cross on stone pedestal, rural Galicia, historic village context"

echo "==> Updating Space variables"
./.venv/bin/python scripts/create_or_update_space.py \
  --space_id "${SPACE_ID}" \
  --space_dir ./space \
  --base_model "${BASE_MODEL}" \
  --lora_target flux \
  --lora_repo_horreo "${MODEL_REPO_HORREO}" \
  --lora_repo_cruceiro "${MODEL_REPO_CRUCEIRO}"

echo "Pipeline complete."
