# Galicia Ethnography Generator (DeepSpeed + Hugging Face)

This project builds a cloud-first workflow to:
- collect images about Galicia ethnography (`horreos` and `cruceiros`),
- train a LoRA on top of `black-forest-labs/FLUX.1-schnell` with `DeepSpeed`,
- run fast inference on `Tongyi-MAI/Z-Image-Turbo`,
- push dataset/model to the Hugging Face Hub,
- deploy a Hugging Face Space for generation.

## Why this avoids local disk usage

You can run everything in ephemeral cloud storage (Colab, Kaggle, or remote GPU VM):
- data is downloaded to `/tmp` or `/content` (ephemeral),
- artifacts are pushed to Hugging Face Hub,
- final inference runs in a Hugging Face Space.

Your laptop only needs this small project repository.

## Important notes

- Google Images content may be copyrighted. Use license filters and verify usage rights before training.
- For safer/legal sourcing, this project includes a Wikimedia collector script.
- `DeepSpeed` accelerates distributed training, but training still needs GPU resources.
- Hugging Face ZeroGPU is suitable for inference demos, not heavy training jobs.
- According to Tongyi-MAI model docs, `Z-Image-Turbo` is designed for fast inference, not fine-tuning.

## Repo layout

- `configs/deepspeed_zero2.json`: DeepSpeed ZeRO Stage 2 config.
- `configs/accelerate_deepspeed_zero2.yaml`: Accelerate config wired to DeepSpeed.
- `scripts/collect_google_images_serpapi.py`: collect Google Images with SerpAPI.
- `scripts/collect_wikimedia_commons.py`: collect from Wikimedia Commons.
- `scripts/build_imagefolder_dataset.py`: create `imagefolder` dataset format.
- `scripts/upload_to_hub.py`: upload local folder to dataset/model/space repo.
- `scripts/train_flux_lora_deepspeed.sh`: launch FLUX LoRA training with DeepSpeed.
- `scripts/create_or_update_space.py`: create and upload a Gradio Space.
- `space/app.py`: generation app (Z-Image-Turbo or FLUX + your LoRA).

## 1) Prerequisites

Create a remote environment (recommended):
- Colab Pro / Kaggle / GPU VM

Install dependencies:

```bash
pip install -r requirements.txt
```

Set credentials:

```bash
export HF_TOKEN="hf_..."
export HF_USERNAME="your_hf_username"
export SERPAPI_API_KEY="..."   # only for Google image collection
```

Login once:

```bash
huggingface-cli login --token "$HF_TOKEN"
```

## 2) Collect images

### Option A (Google Images via SerpAPI)

```bash
python scripts/collect_google_images_serpapi.py \
  --out_dir /tmp/galicia_raw \
  --per_query 120 \
  --queries "horreo galicia" "cruceiro galicia"
```

### Option B (Wikimedia Commons, license-friendly)

```bash
python scripts/collect_wikimedia_commons.py \
  --out_dir /tmp/galicia_raw \
  --per_query 120 \
  --queries "horreo galicia" "cruceiro galicia"
```

## 3) Build a dataset in Hugging Face imagefolder format

```bash
python scripts/build_imagefolder_dataset.py \
  --raw_dir /tmp/galicia_raw \
  --dataset_dir /tmp/galicia_dataset \
  --val_ratio 0.1
```

Upload dataset:

```bash
python scripts/upload_to_hub.py \
  --repo_id "$HF_USERNAME/galicia-ethnography-dataset" \
  --repo_type dataset \
  --folder_path /tmp/galicia_dataset
```

## 4) Train LoRA with DeepSpeed (FLUX script)

Prepare env vars:

```bash
export DATA_DIR=/tmp/galicia_dataset/train
export OUTPUT_DIR=/tmp/flux-galicia-lora
export INSTANCE_PROMPT="ethnographic photo of galician horreo or cruceiro, natural light"
# Keep this launcher on FLUX checkpoints.
# export BASE_MODEL="black-forest-labs/FLUX.1-schnell"
```

Run training:

```bash
bash scripts/train_flux_lora_deepspeed.sh
```

Upload trained LoRA:

```bash
python scripts/upload_to_hub.py \
  --repo_id "$HF_USERNAME/flux-schnell-galicia-lora" \
  --repo_type model \
  --folder_path "$OUTPUT_DIR"
```

## 5) Create and deploy the Space

```bash
python scripts/create_or_update_space.py \
  --space_id "$HF_USERNAME/galicia-horreos-cruceiros" \
  --base_model "Tongyi-MAI/Z-Image-Turbo" \
  --space_dir ./space
```

Then set the Space variable `LORA_REPO` to:
- `$HF_USERNAME/flux-schnell-galicia-lora`

The app in `space/app.py` will load Z-Image-Turbo by default (or FLUX if you set another `BASE_MODEL`).

## 6) Suggested cloud workflow

1. Run collection + dataset build in Colab/Kaggle.
2. Upload dataset to HF Hub.
3. Train on a GPU runner with DeepSpeed.
4. Upload LoRA to HF model repo.
5. Run inference in HF Space.

This gives you near-zero local storage usage and keeps your project fully portable.
