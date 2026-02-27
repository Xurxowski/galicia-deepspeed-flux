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

DATASET_REPO_HORREO="${DATASET_REPO_HORREO:-${HF_USERNAME}/galicia-horreo-dataset-private}"
DATASET_REPO_CRUCEIRO="${DATASET_REPO_CRUCEIRO:-${HF_USERNAME}/galicia-cruceiro-dataset-private}"

MODEL_REPO_HORREO="${MODEL_REPO_HORREO:-${HF_USERNAME}/flux-schnell-galicia-lora-horreo}"
MODEL_REPO_CRUCEIRO="${MODEL_REPO_CRUCEIRO:-${HF_USERNAME}/flux-schnell-galicia-lora-cruceiro}"

OUTPUT_DIR_HORREO="${OUTPUT_DIR_HORREO:-/tmp/flux-galicia-lora-horreo}"
OUTPUT_DIR_CRUCEIRO="${OUTPUT_DIR_CRUCEIRO:-/tmp/flux-galicia-lora-cruceiro}"

MAX_TRAIN_STEPS_HORREO="${MAX_TRAIN_STEPS_HORREO:-500}"
MAX_TRAIN_STEPS_CRUCEIRO="${MAX_TRAIN_STEPS_CRUCEIRO:-700}"
RESOLUTION="${RESOLUTION:-512}"
RANK="${RANK:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
USE_DEEPSPEED="${USE_DEEPSPEED:-auto}"
SKIP_SPACE_UPDATE="${SKIP_SPACE_UPDATE:-0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(
    python - <<'PY'
from huggingface_hub import get_token
print(get_token() or "")
PY
  )"
fi

if [[ -z "${HF_TOKEN}" ]]; then
  echo "Missing HF token. Run: hf auth login or export HF_TOKEN=..." >&2
  exit 1
fi

export HF_TOKEN
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false

run_train_upload() {
  local label="$1"
  local dataset_name="$2"
  local output_dir="$3"
  local model_repo="$4"
  local max_steps="$5"
  local instance_prompt="$6"

  echo "==> [${label}] training start with dataset ${dataset_name}"
  export BASE_MODEL
  export DATASET_NAME="${dataset_name}"
  unset DATA_DIR
  export IMAGE_COLUMN=image
  export CAPTION_COLUMN=text
  export OUTPUT_DIR="${output_dir}"
  export INSTANCE_PROMPT="${instance_prompt}"
  export MAX_TRAIN_STEPS="${max_steps}"
  export RESOLUTION
  export RANK
  export GRAD_ACC
  export LEARNING_RATE
  export MIXED_PRECISION
  export USE_DEEPSPEED
  export CHECKPOINTING_STEPS=100

  bash scripts/train_flux_lora_deepspeed.sh

  echo "==> [${label}] uploading ${model_repo}"
  python scripts/upload_to_hub.py \
    --repo_id "${model_repo}" \
    --repo_type model \
    --folder_path "${output_dir}"
}

run_train_upload \
  "horreo" \
  "${DATASET_REPO_HORREO}" \
  "${OUTPUT_DIR_HORREO}" \
  "${MODEL_REPO_HORREO}" \
  "${MAX_TRAIN_STEPS_HORREO}" \
  "ethnographic documentary photo of a traditional galician horreo (horreo gallego), elongated rectangular raised granary on granite pegollos with circular tornarratos capstones, ventilated slatted stone or wood chamber, gabled tile or slate roof with cruz and pinaculo finials, full exterior visible, weathered granite and moss, rural Galicia, humid atlantic light, no modern house, no interior"

run_train_upload \
  "cruceiro" \
  "${DATASET_REPO_CRUCEIRO}" \
  "${OUTPUT_DIR_CRUCEIRO}" \
  "${MODEL_REPO_CRUCEIRO}" \
  "${MAX_TRAIN_STEPS_CRUCEIRO}" \
  "ethnographic documentary photo of a traditional galician cruceiro de granito, stepped pedestal (gradas), tall monolithic shaft (varal) with carved capital, latin cross with Cristo on front and Virxe on reverse, full monument visible, weathered granite with lichen, by a churchyard or crossroads in rural Galicia, natural overcast atlantic light, no modern sculpture, no generic cemetery cross"

if [[ "${SKIP_SPACE_UPDATE}" != "1" ]]; then
  echo "==> Updating Space variables"
  python scripts/create_or_update_space.py \
    --space_id "${SPACE_ID}" \
    --space_dir ./space \
    --base_model "${BASE_MODEL}" \
    --lora_target flux \
    --lora_repo_horreo "${MODEL_REPO_HORREO}" \
    --lora_repo_cruceiro "${MODEL_REPO_CRUCEIRO}"
fi

echo "Pipeline complete."
