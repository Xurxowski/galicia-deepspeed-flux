#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Usage (GPU machine, Linux + NVIDIA CUDA):
#   export HF_TOKEN="hf_..."
#   export HF_USERNAME="Xurxowsky"
#   export DATASET_NAME="Xurxowsky/galicia-ethnography-dataset-private"
#   export MIXED_PRECISION=fp16   # T4; use bf16 for A10/L4/A100
#   export MAX_TRAIN_STEPS=1200
#   bash scripts/train_and_upload_flux_lora.sh

MODEL_REPO_ID="${MODEL_REPO_ID:-}"
if [[ -z "${MODEL_REPO_ID}" ]]; then
  if [[ -n "${HF_USERNAME:-}" ]]; then
    MODEL_REPO_ID="${HF_USERNAME}/flux-schnell-galicia-lora"
  else
    echo "Missing MODEL_REPO_ID (or HF_USERNAME)." >&2
    echo "Example: export MODEL_REPO_ID='Xurxowsky/flux-schnell-galicia-lora'" >&2
    exit 1
  fi
fi

export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/flux-galicia-lora}"
mkdir -p "${OUTPUT_DIR}"

echo "==> Training FLUX LoRA (output: ${OUTPUT_DIR})"
bash "${PROJECT_ROOT}/scripts/train_flux_lora_deepspeed.sh"

echo "==> Uploading artifacts to Hugging Face: ${MODEL_REPO_ID}"
UPLOAD_ARGS=(--repo_id "${MODEL_REPO_ID}" --repo_type model --folder_path "${OUTPUT_DIR}")
if [[ "${UPLOAD_PRIVATE:-0}" == "1" ]]; then
  UPLOAD_ARGS+=(--private)
fi

python "${PROJECT_ROOT}/scripts/upload_to_hub.py" "${UPLOAD_ARGS[@]}"

echo "==> Done"

