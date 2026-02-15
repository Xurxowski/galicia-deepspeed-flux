---
title: Galicia Horreos Cruceiros Generator
sdk: gradio
app_file: app.py
pinned: false
---

Space focused on Galicia ethnographic elements:
- horreos
- cruceiros

Optional Space variables:
- `BASE_MODEL`: defaults to `Tongyi-MAI/Z-Image-Turbo`
- `LORA_REPO`: your LoRA model repo (example: `username/flux-schnell-galicia-lora`)
- `LORA_WEIGHT_NAME`: LoRA file name (default: `pytorch_lora_weights.safetensors`)

The app auto-detects `BASE_MODEL` family:
- `z-image` -> `ZImagePipeline`
- otherwise -> `FluxPipeline`

If `LORA_REPO` is not set, the app runs with base model only.
