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
- Stock sites (Shutterstock/Adobe Stock/Dreamstime) typically require paid licenses and manual download workflows.
- This repo includes a manual stock import script for already-licensed assets.
- This repo includes a Pixabay API collector for free-license images.
- `DeepSpeed` accelerates distributed training, but training still needs GPU resources.
- Hugging Face ZeroGPU is suitable for inference demos, not heavy training jobs.
- According to Tongyi-MAI model docs, `Z-Image-Turbo` is designed for fast inference, not fine-tuning.

## Repo layout

- `configs/deepspeed_zero2.json`: DeepSpeed ZeRO Stage 2 config.
- `configs/accelerate_deepspeed_zero2.yaml`: Accelerate config wired to DeepSpeed.
- `configs/concepts_galicia.json`: scalable concept map (tokens + captions) for multi-concept LoRA.
- `scripts/collect_google_images_serpapi.py`: collect Google Images with SerpAPI.
- `scripts/collect_wikimedia_commons.py`: collect from Wikimedia Commons.
- `scripts/collect_pixabay_images.py`: collect from Pixabay API.
- `scripts/collect_cruceirosdegalicia.py`: conservative collector for cruceirosdegalicia.xyz (requires explicit rights confirmation).
- `scripts/collect_horreosdegalicia.py`: conservative collector for horreosdegalicia.com (requires explicit rights confirmation).
- `scripts/import_licensed_stock_manual.py`: import manually licensed stock photos.
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
export PIXABAY_API_KEY="..."   # only for Pixabay collection
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

This writes a traceability file: `/tmp/galicia_raw/sources_wikimedia_commons.jsonl`.

### Option C (Pixabay API, free-license workflow)

```bash
python scripts/collect_pixabay_images.py \
  --out_dir /tmp/galicia_raw \
  --per_query 200 \
  --queries "horreo gallego" "horreo galicia"
```

### Option D (Licensed stock photos, manual import)

Download assets manually from your licensed account first, then import:

```bash
python scripts/import_licensed_stock_manual.py \
  --input_dir /tmp/stock_downloads/horreos \
  --out_dir /tmp/galicia_raw \
  --label horreo \
  --source shutterstock \
  --license_reference "invoice-2026-02-15"
```

### Option E (Cruceirosdegalicia.xyz, only with explicit permission)

This site aggregates photos from many contributors. Use this only if you have explicit rights for ML training usage.
Contact shown on site metadata: `cruceirosgalicia@gmail.com`.

```bash
python scripts/collect_cruceirosdegalicia.py \
  --out_dir /tmp/galicia_raw \
  --max_images 1500 \
  --sleep_sec 0.8 \
  --confirm_rights I_HAVE_PERMISSION
```

Recommended starting volume for cruceiros:
- first round: 800-1500 photos
- second round (if needed): up to 2500 photos

You usually do not need all 38k photos for a good LoRA.

### Option F (Horreosdegalicia.com, only with explicit permission)

Use this only if you have explicit rights for ML training usage.

```bash
python scripts/collect_horreosdegalicia.py \
  --out_dir /tmp/galicia_raw \
  --max_images 800 \
  --sleep_sec 0.8 \
  --confirm_rights I_HAVE_PERMISSION
```

## 3) Build a dataset in Hugging Face imagefolder format

```bash
python scripts/build_imagefolder_dataset.py \
  --raw_dir /tmp/galicia_raw \
  --dataset_dir /tmp/galicia_dataset \
  --val_ratio 0.1 \
  --caption_profile horreo_focus \
  --include_labels horreo
```

Upload dataset:

```bash
python scripts/upload_to_hub.py \
  --repo_id "$HF_USERNAME/galicia-ethnography-dataset" \
  --repo_type dataset \
  --folder_path /tmp/galicia_dataset
```

If you are still confirming permissions/licenses for some sources, upload to a private repo first:

```bash
python scripts/upload_to_hub.py \
  --repo_id "$HF_USERNAME/galicia-ethnography-dataset-private" \
  --repo_type dataset \
  --folder_path /tmp/galicia_dataset \
  --private
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

Important:
- This training path requires **Linux + NVIDIA GPU (CUDA)**. `DeepSpeed` generally will not work on macOS.
- For T4 GPUs, use `MIXED_PRECISION=fp16`.

Run training:

```bash
export MIXED_PRECISION=bf16   # use fp16 on T4
bash scripts/train_flux_lora_deepspeed.sh
```

Upload trained LoRA:

```bash
python scripts/upload_to_hub.py \
  --repo_id "$HF_USERNAME/flux-schnell-galicia-lora" \
  --repo_type model \
  --folder_path "$OUTPUT_DIR"
```

### One-command: train + upload

This wrapper sets a project-local output folder and uploads at the end:

```bash
export MODEL_REPO_ID="$HF_USERNAME/flux-schnell-galicia-lora"
export DATASET_NAME="$HF_USERNAME/galicia-ethnography-dataset-private"
export MIXED_PRECISION=fp16   # T4; use bf16 for A10/L4/A100
bash scripts/train_and_upload_flux_lora.sh
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

## 7) Facebook Group + Concept-Specific Retraining

If you use Facebook groups (for example, `TODOS LOS HORREOS DE GALICIA RECOPILADOS`), do not scrape automatically.
Use manual download only, and keep permission/license references.

Important:
- the current FLUX DreamBooth launcher uses one `INSTANCE_PROMPT` for all images.
- for better fidelity, train one LoRA for `horreo` and another LoRA for `cruceiro`.

### A) Import manually downloaded Facebook images

Horreos:

```bash
python scripts/import_licensed_stock_manual.py \
  --input_dir /tmp/facebook_downloads/horreos \
  --out_dir /tmp/galicia_raw \
  --label horreo \
  --source facebook_group_todos_horreos_galicia \
  --license_reference "fb-group-permission-2026-02-15"
```

## 8) Use the LoRA in Draw Things (macOS)

Once your model repo contains a `*.safetensors` LoRA (typically `pytorch_lora_weights.safetensors`), download it and copy into Draw Things Downloads:

```bash
python scripts/download_lora_for_drawthings.py \
  --repo_id "$HF_USERNAME/flux-schnell-galicia-lora" \
  --copy_to_drawthings_downloads
```

Then import it in Draw Things: LoRA -> Manage/Import -> Local file.

Cruceiros:

```bash
python scripts/import_licensed_stock_manual.py \
  --input_dir /tmp/facebook_downloads/cruceiros \
  --out_dir /tmp/galicia_raw \
  --label cruceiro \
  --source facebook_group_cruceiros_galicia \
  --license_reference "fb-group-permission-2026-02-15"
```

### B) Build concept-specific datasets

Horreo dataset:

```bash
python scripts/build_imagefolder_dataset.py \
  --raw_dir /tmp/galicia_raw \
  --dataset_dir /tmp/galicia_dataset_horreo \
  --caption_profile horreo_focus \
  --include_labels horreo \
  --val_ratio 0.1
```

Cruceiro dataset:

```bash
python scripts/build_imagefolder_dataset.py \
  --raw_dir /tmp/galicia_raw \
  --dataset_dir /tmp/galicia_dataset_cruceiro \
  --caption_profile cruceiro_focus \
  --include_labels cruceiro \
  --val_ratio 0.1
```

### C) Train one LoRA per concept

Horreo LoRA:

```bash
export DATA_DIR=/tmp/galicia_dataset_horreo/train
export OUTPUT_DIR=/tmp/flux-galicia-lora-horreo
export INSTANCE_PROMPT="ethnographic documentary photo of a traditional galician horreo, raised granary on stone pillars (pegollos), elongated slatted chamber, Galicia"
bash scripts/train_flux_lora_deepspeed.sh
```

Cruceiro LoRA:

```bash
export DATA_DIR=/tmp/galicia_dataset_cruceiro/train
export OUTPUT_DIR=/tmp/flux-galicia-lora-cruceiro
export INSTANCE_PROMPT="ethnographic documentary photo of a galician cruceiro, carved granite cross on stone pedestal, historic village context, Galicia"
bash scripts/train_flux_lora_deepspeed.sh
```

### D) Upload both LoRAs

```bash
python scripts/upload_to_hub.py \
  --repo_id "$HF_USERNAME/flux-schnell-galicia-lora-horreo" \
  --repo_type model \
  --folder_path /tmp/flux-galicia-lora-horreo

python scripts/upload_to_hub.py \
  --repo_id "$HF_USERNAME/flux-schnell-galicia-lora-cruceiro" \
  --repo_type model \
  --folder_path /tmp/flux-galicia-lora-cruceiro
```

## 8) Single LoRA with Differentiated Concepts (Scalable)

If you prefer one LoRA with clearly separated concepts (`horreo`, `cruceiro`, and future `muino`), use concept tokens and per-image captions.

### A) Put images by label folder

```text
/tmp/galicia_raw/
  horreo/*.jpg
  cruceiro/*.jpg
  muino/*.jpg   # optional for future
```

### B) Build multi-concept dataset with concept tokens

```bash
python scripts/build_imagefolder_dataset.py \
  --raw_dir /tmp/galicia_raw \
  --dataset_dir /tmp/galicia_dataset_multiconcept \
  --val_ratio 0.1 \
  --concepts_file configs/concepts_galicia.json \
  --use_concept_tokens \
  --balance \
  --include_labels horreo cruceiro
```

You can later add `muino` by adding images to `raw_dir/muino` and including it:

```bash
python scripts/build_imagefolder_dataset.py \
  --raw_dir /tmp/galicia_raw \
  --dataset_dir /tmp/galicia_dataset_multiconcept \
  --val_ratio 0.1 \
  --concepts_file configs/concepts_galicia.json \
  --use_concept_tokens \
  --include_labels horreo cruceiro muino
```

### C) Train single multi-concept LoRA using caption column

```bash
export DATASET_NAME=/tmp/galicia_dataset_multiconcept/train
export IMAGE_COLUMN=image
export CAPTION_COLUMN=text
export OUTPUT_DIR=/tmp/flux-galicia-lora-multiconcept
export INSTANCE_PROMPT=\"ethnographic photo of galician traditional architecture\"
bash scripts/train_flux_lora_deepspeed.sh
```

### D) Prompting with concept tokens

Use the concept token in generation prompts for stronger separation:
- `horreo`: `<gal_horreo>`
- `cruceiro`: `<gal_cruceiro>`
- `muino`: `<gal_muino>`
